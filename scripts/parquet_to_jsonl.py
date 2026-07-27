#!/usr/bin/env python3
"""
Parquet to JSONL 转换脚本（含媒体文件恢复）

将 Parquet 文件转换回 JSONL 格式，同时从 bytes 恢复媒体文件。
- audio_bytes → WavPath 对应的音频文件（或视频同名 wav）
- image_bytes → ImagePath 对应的图片文件
- 视频不在 Parquet 中，需用户单独下载

支持分片 Parquet 输入（data-00000-of-00003.parquet 格式）。

用法:
    # 测试模式
    python parquet_to_jsonl.py --test
    
    # 完整转换
    python parquet_to_jsonl.py --input_dir ../data_parquet --output_dir ../data_restored
    
    # 指定单个文件（也支持分片 glob）
    python parquet_to_jsonl.py --input_file data.parquet --output_file data.jsonl --media_output_dir ./media
    
    # 指定 data_prefix_dir，让媒体还原到指定目录
    python parquet_to_jsonl.py --input_file data.parquet --output_file data.jsonl --data_prefix_dir ./data/omni/raw_hf/omnibench/
"""

import os
import sys
import json
import glob as glob_module
import argparse
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import pandas as pd
    import pyarrow.parquet as pq
except ImportError:
    print("请先安装依赖: pip install pandas pyarrow")
    sys.exit(1)


def try_parse_json(value: str) -> Any:
    """尝试将字符串解析为 JSON，失败则返回原值"""
    if not isinstance(value, str):
        return value
    
    stripped = value.strip()
    if not stripped:
        return value
    
    if (stripped.startswith('{') and stripped.endswith('}')) or \
       (stripped.startswith('[') and stripped.endswith(']')):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
    
    return value


def normalize_missing_value(value: Any) -> Any:
    try:
        return None if bool(pd.isna(value)) else value
    except (TypeError, ValueError):
        return value


def write_file_bytes(file_path: str, data: bytes) -> bool:
    """写入 bytes 到文件
    
    Args:
        file_path: 文件路径
        data: bytes 数据
    
    Returns:
        是否成功
    """
    try:
        os.makedirs(os.path.dirname(file_path) or '.', exist_ok=True)
        with open(file_path, 'wb') as f:
            f.write(data)
        return True
    except Exception as e:
        print(f"警告: 无法写入文件 {file_path}: {e}")
        return False


def read_parquet_file(path: str, drop_media: bool = False) -> pd.DataFrame:
    if not drop_media:
        return pd.read_parquet(path)

    columns = [
        name for name in pq.ParquetFile(path).schema_arrow.names
        if name not in ('audio', 'image', 'video')
        and not name.endswith('_bytes')
        and not name.endswith('_bytes_dict')
    ]
    return pd.read_parquet(path, columns=columns)


def read_parquet_sharded(input_path: str, drop_media: bool = False) -> pd.DataFrame:
    """读取 Parquet 文件，支持分片格式。
    
    如果 input_path 是单个文件则直接读取；
    如果目录下存在 data-*-of-*.parquet 分片文件则合并读取。
    """
    if os.path.isfile(input_path):
        base, ext = os.path.splitext(input_path)
        shard_pattern = f"{base}-*-of-*{ext}"
        shard_files = sorted(glob_module.glob(shard_pattern))
        if shard_files:
            dfs = [read_parquet_file(f, drop_media=drop_media) for f in shard_files]
            return pd.concat(dfs, ignore_index=True)
        return read_parquet_file(input_path, drop_media=drop_media)
    
    if os.path.isdir(input_path):
        pattern = os.path.join(input_path, "*.parquet")
        files = sorted(glob_module.glob(pattern))
        if not files:
            raise FileNotFoundError(f"目录中没有 Parquet 文件: {input_path}")
        dfs = [read_parquet_file(f, drop_media=drop_media) for f in files]
        return pd.concat(dfs, ignore_index=True)
    
    shard_pattern = input_path.replace('.parquet', '-*-of-*.parquet')
    shard_files = sorted(glob_module.glob(shard_pattern))
    if shard_files:
        dfs = [read_parquet_file(f, drop_media=drop_media) for f in shard_files]
        return pd.concat(dfs, ignore_index=True)
    
    raise FileNotFoundError(f"找不到 Parquet 文件: {input_path}")


def parquet_to_jsonl(
    input_path: str, 
    output_path: str, 
    media_output_dir: str,
    limit: Optional[int] = None,
    restore_media: bool = True,
    data_prefix_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """将 Parquet 文件转换为 JSONL 格式（恢复媒体文件）
    
    Args:
        input_path: 输入 Parquet 文件路径（支持分片目录或单文件）
        output_path: 输出 JSONL 文件路径
        media_output_dir: 媒体文件输出目录（默认行为）
        limit: 最大转换条数
        restore_media: 是否恢复媒体文件
        data_prefix_dir: 如果指定，媒体文件还原到此目录（覆盖 media_output_dir）
    
    Returns:
        转换统计信息
    """
    effective_media_dir = data_prefix_dir or media_output_dir
    df = read_parquet_sharded(input_path, drop_media=not restore_media)
    
    if limit is not None:
        df = df.head(limit)
    
    if len(df) == 0:
        return {"status": "empty", "input": input_path, "count": 0}
    
    stats = {
        "audio_restored": 0,
        "image_restored": 0,
    }
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, row in df.iterrows():
            record = row.to_dict()
            
            # 处理 NaN 值
            record = {k: normalize_missing_value(v) for k, v in record.items()}
            
            if restore_media:
                audio_data = None
                audio_path = None

                # HF struct 格式: audio = {"bytes": b'...', "path": "xxx.wav"}
                if 'audio' in record and isinstance(record.get('audio'), dict):
                    audio_struct = record['audio']
                    audio_data = audio_struct.get('bytes')
                    audio_path = audio_struct.get('path') or record.get('WavPath') or record.get('audio_path')
                    del record['audio']
                # 旧格式: audio_bytes = b'...'
                elif 'audio_bytes' in record and record['audio_bytes'] is not None:
                    audio_data = record['audio_bytes']
                    audio_path = record.get('WavPath') or record.get('audio_path')
                    del record['audio_bytes']

                if audio_data is not None:
                    if not audio_path:
                        video_path = record.get('VideoPath') or record.get('video_path')
                        if video_path:
                            audio_path = os.path.splitext(video_path)[0] + '.wav'
                    if audio_path:
                        full_path = os.path.join(effective_media_dir, audio_path)
                        if write_file_bytes(full_path, audio_data):
                            stats["audio_restored"] += 1

                image_data = None
                image_path = None

                # HF struct 格式: image = {"bytes": b'...', "path": "xxx.png"}
                if 'image' in record and isinstance(record.get('image'), dict):
                    image_struct = record['image']
                    image_data = image_struct.get('bytes')
                    image_path = image_struct.get('path') or record.get('ImagePath') or record.get('image_path')
                    del record['image']
                # 旧格式: image_bytes = b'...'
                elif 'image_bytes' in record and record['image_bytes'] is not None:
                    image_data = record['image_bytes']
                    image_path = record.get('ImagePath') or record.get('image_path')
                    del record['image_bytes']

                if image_data is not None and image_path:
                    full_path = os.path.join(effective_media_dir, image_path)
                    if write_file_bytes(full_path, image_data):
                        stats["image_restored"] += 1

                for bytes_field, path_field in [
                    ('audio_bytes_dict', 'audio_paths_dict'),
                    ('image_bytes_dict', 'image_paths_dict'),
                ]:
                    if bytes_field in record and record[bytes_field]:
                        try:
                            import base64
                            encoded_dict = json.loads(record[bytes_field])
                            paths_dict = record.get(path_field)
                            if isinstance(paths_dict, str):
                                paths_dict = json.loads(paths_dict)
                            
                            if paths_dict:
                                for key, encoded_data in encoded_dict.items():
                                    if key in paths_dict:
                                        file_bytes = base64.b64decode(encoded_data)
                                        full_path = os.path.join(effective_media_dir, paths_dict[key])
                                        write_file_bytes(full_path, file_bytes)
                        except Exception as e:
                            print(f"警告: 恢复 {bytes_field} 失败: {e}")
                        del record[bytes_field]
            
            # 移除所有媒体字段
            record = {k: v for k, v in record.items()
                      if k not in ('audio', 'image', 'video')
                      and not k.endswith('_bytes') and not k.endswith('_bytes_dict')}
            
            # 尝试恢复嵌套结构
            record = {k: try_parse_json(v) for k, v in record.items()}
            
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False,
                    default=lambda b: b.decode("utf-8", errors="replace") if isinstance(b, bytes) else str(b),
                )
                + '\n'
            )
    
    try:
        input_size = os.path.getsize(input_path) if os.path.isfile(input_path) else 0
    except OSError:
        input_size = 0
    output_size = os.path.getsize(output_path)
    
    return {
        "status": "success",
        "input": input_path,
        "output": output_path,
        "media_dir": effective_media_dir,
        "count": len(df),
        "columns": [c for c in df.columns if not c.endswith('_bytes')],
        "input_size_mb": round(input_size / 1024 / 1024, 2),
        "output_size_mb": round(output_size / 1024 / 1024, 2),
        "stats": stats
    }


def find_parquet_files(root_dir: str) -> List[str]:
    """递归查找所有 Parquet 文件"""
    parquet_files = []
    root_path = Path(root_dir)
    
    for path in root_path.rglob("*.parquet"):
        try:
            if path.exists() and path.is_file():
                parquet_files.append(str(path))
        except Exception as e:
            print(f"警告: 无法解析路径 {path}: {e}")
    
    return parquet_files


def convert_directory(
    input_dir: str, 
    output_dir: str, 
    limit: Optional[int] = None,
    restore_media: bool = True
) -> List[Dict[str, Any]]:
    """转换目录下的所有 Parquet 文件"""
    results = []
    
    parquet_files = find_parquet_files(input_dir)
    print(f"找到 {len(parquet_files)} 个 Parquet 文件")
    
    for parquet_path in parquet_files:
        rel_path = os.path.relpath(parquet_path, input_dir)
        output_path = os.path.join(output_dir, rel_path.replace('.parquet', '.jsonl'))
        media_output_dir = os.path.dirname(output_path)
        
        print(f"转换: {rel_path}")
        try:
            result = parquet_to_jsonl(parquet_path, output_path, media_output_dir, limit, restore_media)
            results.append(result)
            if result["status"] == "success":
                stats = result.get("stats", {})
                print(f"  ✓ {result['count']} 条")
                if restore_media:
                    print(f"    恢复: 音频 {stats.get('audio_restored', 0)}, 图片 {stats.get('image_restored', 0)}")
        except Exception as e:
            print(f"  ✗ 失败: {e}")
            import traceback
            traceback.print_exc()
            results.append({"status": "error", "input": parquet_path, "error": str(e)})
    
    return results


def test_conversion():
    """测试模式"""
    print("=" * 60)
    print("测试模式: Parquet → JSONL + 媒体文件恢复")
    print("=" * 60)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(script_dir, "../data_parquet_test")
    output_dir = os.path.join(script_dir, "../data_jsonl_restored_test")
    
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录不存在: {input_dir}")
        print("请先运行: python jsonl_to_parquet.py --test")
        return []
    
    import shutil
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    
    results = convert_directory(input_dir, output_dir, limit=10, restore_media=True)
    
    success = sum(1 for r in results if r.get("status") == "success")
    empty = sum(1 for r in results if r.get("status") == "empty")
    error = sum(1 for r in results if r.get("status") == "error")
    
    total_audio = sum(r.get("stats", {}).get("audio_restored", 0) for r in results if r.get("status") == "success")
    total_image = sum(r.get("stats", {}).get("image_restored", 0) for r in results if r.get("status") == "success")
    
    print("\n" + "=" * 60)
    print(f"测试完成: 成功 {success}, 空文件 {empty}, 失败 {error}")
    print(f"媒体恢复: 音频 {total_audio}, 图片 {total_image}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Parquet to JSONL 转换工具（含媒体恢复，支持分片输入）")
    parser.add_argument("--test", action="store_true", help="测试模式")
    parser.add_argument("--input_dir", type=str, help="输入目录")
    parser.add_argument("--output_dir", type=str, help="输出目录")
    parser.add_argument("--input_file", type=str, help="单个输入文件（支持分片目录）")
    parser.add_argument("--output_file", type=str, help="单个输出文件")
    parser.add_argument("--media_output_dir", type=str, help="媒体文件输出目录")
    parser.add_argument("--data_prefix_dir", type=str, help="指定媒体还原根目录（覆盖 media_output_dir）")
    parser.add_argument("--limit", type=int, default=None, help="每个文件最大转换条数")
    parser.add_argument("--no_media", action="store_true", help="不恢复媒体文件")
    
    args = parser.parse_args()
    
    if args.test:
        test_conversion()
    elif args.input_file and args.output_file:
        media_dir = args.media_output_dir or os.path.dirname(args.output_file)
        result = parquet_to_jsonl(
            args.input_file, args.output_file, media_dir,
            args.limit, not args.no_media, args.data_prefix_dir,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    elif args.input_dir and args.output_dir:
        results = convert_directory(args.input_dir, args.output_dir, args.limit, not args.no_media)
        success = sum(1 for r in results if r.get("status") == "success")
        print(f"\n完成: {success}/{len(results)} 个文件转换成功")
    else:
        parser.print_help()
        print("\n示例:")
        print("  python parquet_to_jsonl.py --test")
        print("  python parquet_to_jsonl.py --input_dir ../data_parquet --output_dir ../data_restored")
        print("  python parquet_to_jsonl.py --input_file data.parquet --output_file data.jsonl --data_prefix_dir ./data/omni/raw_hf/omnibench/")


if __name__ == "__main__":
    main()
