#!/usr/bin/env python3
"""
OmniEvalKit HuggingFace 数据集下载 & 还原脚本

从 HuggingFace 仓库下载 Parquet 数据集并还原为本地 data/ 目录结构，
使得框架可以直接使用。

用法:
    # 下载全部数据集（音频+图片，不含视频）
    python scripts/hf_download.py --output_dir ./data

    # 只下载指定数据集
    python scripts/hf_download.py --datasets omnibench,daily_omni --output_dir ./data

    # 下载并获取视频
    python scripts/hf_download.py --output_dir ./data --download_videos

    # 查看可用数据集列表
    python scripts/hf_download.py --list
"""

import os
import sys
import json
import argparse
import shutil
import tempfile
from typing import Dict, List, Any, Optional

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, SCRIPT_DIR)

DEFAULT_REPO_ID = "OmniEvalKit/omnievalkit-dataset"


def get_hf_api():
    try:
        from huggingface_hub import HfApi, hf_hub_download, snapshot_download
        return HfApi(), hf_hub_download, snapshot_download
    except ImportError:
        print("错误: 请安装 huggingface_hub: pip install huggingface_hub")
        sys.exit(1)


def download_dataset_info(repo_id: str) -> Dict[str, Any]:
    """下载并解析 dataset_info.json"""
    _, hf_hub_download, _ = get_hf_api()
    
    info_path = hf_hub_download(
        repo_id=repo_id,
        filename="dataset_info.json",
        repo_type="dataset",
    )
    with open(info_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def list_datasets(info: Dict[str, Any]):
    """列出所有可用数据集"""
    datasets = info.get("datasets", {})
    
    print(f"仓库: {info.get('repo_id', '?')}")
    print(f"版本: {info.get('version', '?')}")
    print(f"数据集总数: {info.get('total_datasets', len(datasets))}")
    print()
    
    by_category = {}
    for name, ds in datasets.items():
        cat = ds.get("category", "other")
        by_category.setdefault(cat, []).append(ds)
    
    for cat in sorted(by_category.keys()):
        ds_list = by_category[cat]
        print(f"【{cat}】({len(ds_list)} 个)")
        for ds in sorted(ds_list, key=lambda x: x.get("name", "")):
            video_tag = " [需下载视频]" if ds.get("has_video") else ""
            size = ds.get("size_mb", 0)
            size_str = f"{size:.0f}MB" if size < 1024 else f"{size/1024:.1f}GB"
            print(f"  {ds['name']:<35s} {ds.get('num_samples', '?'):>6} 样本  {size_str:>8}{video_tag}")
        print()


def download_and_restore_dataset(
    ds_info: Dict[str, Any],
    repo_id: str,
    output_dir: str,
):
    """下载单个数据集的 Parquet 并还原"""
    from parquet_to_jsonl import parquet_to_jsonl
    _, hf_hub_download, _ = get_hf_api()
    
    name = ds_info["name"]
    hf_path = ds_info["hf_path"]
    num_shards = ds_info.get("num_shards", 1)
    restore_to = ds_info.get("restore_to", {})
    
    annotation_path = restore_to.get("annotation_path", "")
    data_prefix_dir = restore_to.get("data_prefix_dir", "")
    
    project_root = os.path.abspath(os.path.join(output_dir, ".."))
    
    def _resolve_restore_path(path: str, hf_path: str, is_annotation: bool) -> str:
        """将 restore_to 路径解析为绝对路径，兼容旧版绝对路径"""
        if not path:
            return ""
        if path.startswith("./"):
            path = path[2:]
        if os.path.isabs(path):
            # 旧版 dataset_info.json 中可能残留绝对路径，根据 hf_path 推断正确的相对路径
            basename = os.path.basename(path)
            if is_annotation:
                path = os.path.join("data", hf_path, basename)
            else:
                path = os.path.join("data", hf_path)
            print(f"  警告: restore_to 中包含绝对路径，已自动修正为 {path}")
        return os.path.normpath(os.path.join(project_root, path))
    
    abs_annotation = _resolve_restore_path(annotation_path, hf_path, is_annotation=True)
    abs_data_prefix = _resolve_restore_path(data_prefix_dir, hf_path, is_annotation=False)
    if not abs_data_prefix:
        abs_data_prefix = os.path.abspath(output_dir)
    
    if os.path.exists(abs_annotation):
        print(f"  跳过（已存在）: {abs_annotation}")
        return {"name": name, "status": "skipped"}
    
    tmp_dir = tempfile.mkdtemp(prefix=f"hf_{name}_")
    
    try:
        split = ds_info.get("split", "test")
        downloaded_files = []
        for i in range(num_shards):
            shard_name = f"{hf_path}/{split}-{i:05d}-of-{num_shards:05d}.parquet"
            local = hf_hub_download(
                repo_id=repo_id,
                filename=shard_name,
                repo_type="dataset",
            )
            downloaded_files.append(local)
        
        if len(downloaded_files) > 1:
            merged_dir = os.path.join(tmp_dir, "shards")
            os.makedirs(merged_dir, exist_ok=True)
            for f in downloaded_files:
                shutil.copy2(f, merged_dir)
            parquet_input = merged_dir
        else:
            parquet_input = downloaded_files[0]
        
        os.makedirs(os.path.dirname(abs_annotation) or '.', exist_ok=True)
        os.makedirs(abs_data_prefix, exist_ok=True)
        
        result = parquet_to_jsonl(
            input_path=parquet_input,
            output_path=abs_annotation,
            media_output_dir=abs_data_prefix,
            restore_media=True,
            data_prefix_dir=abs_data_prefix,
        )
        
        return {
            "name": name,
            "status": "success",
            "annotation_path": abs_annotation,
            "data_prefix_dir": abs_data_prefix,
            "count": result.get("count", 0),
            "stats": result.get("stats", {}),
        }
    
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"name": name, "status": "error", "error": str(e)}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def download_videos_for_dataset(ds_info: Dict[str, Any], output_dir: str):
    """尝试下载视频"""
    video_source = ds_info.get("video_source")
    if not video_source:
        print(f"  无视频来源信息，请手动下载")
        return
    
    src_type = video_source.get("type", "")
    
    if src_type == "hf_dataset":
        repo_id = video_source.get("repo_id", "")
        instructions = video_source.get("instructions", "")
        print(f"  视频来源: {repo_id}")
        print(f"  操作说明: {instructions}")
        
        try:
            _, _, snapshot_download = get_hf_api()
            
            restore_to = ds_info.get("restore_to", {})
            data_prefix_dir = restore_to.get("data_prefix_dir", "")
            if data_prefix_dir.startswith("./"):
                data_prefix_dir = data_prefix_dir[2:]
            project_root = os.path.abspath(os.path.join(output_dir, ".."))
            if os.path.isabs(data_prefix_dir):
                hf_path = ds_info.get("hf_path", "")
                data_prefix_dir = os.path.join("data", hf_path)
            target_dir = os.path.normpath(os.path.join(project_root, data_prefix_dir))
            
            print(f"  正在从 {repo_id} 下载视频到 {target_dir} ...")
            snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                local_dir=target_dir,
                ignore_patterns=["*.parquet", "*.jsonl", "*.json", "*.md"],
            )
            print(f"  视频下载完成")
        except Exception as e:
            print(f"  自动下载失败: {e}")
            print(f"  请手动执行: {instructions}")
    else:
        instructions = video_source.get("instructions", "请参考数据集文档手动下载")
        print(f"  {instructions}")


def main():
    parser = argparse.ArgumentParser(description="OmniEvalKit 数据集下载 & 还原")
    parser.add_argument("--repo_id", type=str, default=DEFAULT_REPO_ID,
                        help=f"HuggingFace 仓库 ID（默认 {DEFAULT_REPO_ID}）")
    parser.add_argument("--output_dir", type=str, default="./data",
                        help="输出根目录（默认 ./data）")
    parser.add_argument("--datasets", type=str, default=None,
                        help="逗号分隔的数据集名称")
    parser.add_argument("--download_videos", action="store_true",
                        help="尝试从原始来源下载视频文件")
    parser.add_argument("--list", action="store_true", dest="list_only",
                        help="仅列出可用数据集，不下载")
    args = parser.parse_args()
    
    print(f"正在获取数据集信息: {args.repo_id}")
    info = download_dataset_info(args.repo_id)
    
    if args.list_only:
        list_datasets(info)
        return
    
    all_datasets = info.get("datasets", {})
    
    if args.datasets:
        names = set(args.datasets.split(","))
        selected = {k: v for k, v in all_datasets.items() if k in names}
        missing = names - set(selected.keys())
        if missing:
            print(f"警告: 以下数据集未找到: {', '.join(missing)}")
    else:
        selected = all_datasets
    
    if not selected:
        print("没有匹配的数据集")
        return
    
    print("=" * 60)
    print(f"OmniEvalKit 数据集下载")
    print(f"仓库: {args.repo_id}")
    print(f"输出目录: {args.output_dir}")
    print(f"选中数据集: {len(selected)}")
    print("=" * 60)
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    success_count = 0
    skip_count = 0
    fail_count = 0
    video_needed = []
    
    for i, (name, ds_info) in enumerate(sorted(selected.items())):
        print(f"\n[{i+1}/{len(selected)}] {ds_info.get('display_name', name)} ({name})")
        
        result = download_and_restore_dataset(ds_info, args.repo_id, args.output_dir)
        status = result.get("status")
        
        if status == "success":
            success_count += 1
            print(f"  完成: {result.get('count', '?')} 样本 → {result.get('annotation_path', '')}")
        elif status == "skipped":
            skip_count += 1
        else:
            fail_count += 1
            print(f"  失败: {result.get('error', 'unknown')}")
        
        if ds_info.get("has_video"):
            video_needed.append(ds_info)
    
    print("\n" + "=" * 60)
    print("下载汇总")
    print("=" * 60)
    print(f"成功: {success_count}")
    print(f"跳过（已存在）: {skip_count}")
    print(f"失败: {fail_count}")
    
    if video_needed:
        print(f"\n需要额外下载视频的数据集 ({len(video_needed)}):")
        for ds in video_needed:
            src = ds.get("video_source")
            hint = f" → {src['repo_id']}" if src and src.get("repo_id") else ""
            print(f"  - {ds['name']}{hint}")
        
        if args.download_videos:
            print("\n正在下载视频...")
            for ds in video_needed:
                print(f"\n下载视频: {ds['name']}")
                download_videos_for_dataset(ds, args.output_dir)
        else:
            print(f"\n提示: 使用 --download_videos 自动下载视频")
    
    print(f"\n数据已还原到: {os.path.abspath(args.output_dir)}")
    print("现在可以运行评测了！")


if __name__ == "__main__":
    main()
