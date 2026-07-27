#!/usr/bin/env python3
import argparse
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
from huggingface_hub import hf_hub_download, snapshot_download


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw_hf_vlm"


DATASETS = {
    "chartqa",
    "ocrbench",
    "textvqa",
    "docvqa",
    "infographicvqa",
    "mathvista",
    "mathverse_mini",
    "videomme",
    "videomme_short",
    "lvbench",
    "longvideobench_val",
    "motionbench",
    "pixmo_count",
    "pixmo_pointing",
    "refcoco",
    "mmbench",
    "mme",
    "mmstar",
    "mvbench",
    "egoschema",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets", nargs="+", choices=sorted(DATASETS) + ["all"])
    parser.add_argument("--skip-large-videos", action="store_true", help="Prepare metadata for large video datasets without downloading video payloads.")
    args = parser.parse_args()

    selected = sorted(DATASETS) if "all" in args.datasets else args.datasets
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    for name in selected:
        print(f"Preparing {name}...")
        PREPARE[name](skip_large_videos=args.skip_large_videos)


def prepare_chartqa(skip_large_videos=False):
    rows = _load_parquets("lmms-lab/ChartQA", ["data/test-00000-of-00001.parquet"], "chartqa")
    out = DATA_ROOT / "chartqa"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        image_path = _write_image(row.get("image"), out / "images", f"chartqa_{idx:06d}")
        records.append(_record(f"chartqa_{idx:06d}", row.get("question"), row.get("answer"), image_path=image_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_ocrbench(skip_large_videos=False):
    rows = _load_parquets("echo840/OCRBench", ["data/test-00000-of-00001.parquet"], "ocrbench")
    out = DATA_ROOT / "ocrbench"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        image_path = _write_image(row.get("image"), out / "images", f"ocrbench_{idx:06d}")
        answers = _as_list(row.get("answer"))
        records.append(_record(f"ocrbench_{idx:06d}", row.get("question"), answers[0] if answers else "", references=answers, image_path=image_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_textvqa(skip_large_videos=False):
    files = [f"data/validation-{idx:05d}-of-00003.parquet" for idx in range(3)]
    rows = _load_parquets("lmms-lab/textvqa", files, "textvqa")
    out = DATA_ROOT / "textvqa"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("question_id") or f"textvqa_{idx:06d}")
        image_path = _write_image(row.get("image"), out / "images", f"chartqa_{idx:06d}")
        answers = _as_list(row.get("answers"))
        records.append(_record(sample_id, row.get("question"), answers[0] if answers else "", references=answers, image_path=image_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_docvqa(skip_large_videos=False):
    files = [f"DocVQA/validation-{idx:05d}-of-00006.parquet" for idx in range(6)]
    rows = _load_parquets("lmms-lab/DocVQA", files, "docvqa")
    out = DATA_ROOT / "docvqa"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("questionId") or f"docvqa_{idx:06d}")
        image_path = _write_image(row.get("image"), out / "images", sample_id)
        answers = _as_list(row.get("answers"))
        records.append(_record(sample_id, row.get("question"), answers[0] if answers else "", references=answers, image_path=image_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_infographicvqa(skip_large_videos=False):
    files = [f"InfographicVQA/validation-{idx:05d}-of-00004.parquet" for idx in range(4)]
    rows = _load_parquets("lmms-lab/DocVQA", files, "infographicvqa")
    out = DATA_ROOT / "infographicvqa"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("questionId") or f"infographicvqa_{idx:06d}")
        image_path = _write_image(row.get("image"), out / "images", sample_id)
        answers = _as_list(row.get("answers"))
        records.append(_record(sample_id, row.get("question"), answers[0] if answers else "", references=answers, image_path=image_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_mathvista(skip_large_videos=False):
    rows = _load_parquets("AI4Math/MathVista", ["data/testmini-00000-of-00001-725687bf7a18d64b.parquet"], "mathvista")
    out = DATA_ROOT / "mathvista"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("pid") or f"mathvista_{idx:06d}")
        image_path = _write_image(row.get("decoded_image"), out / "images", sample_id)
        question = row.get("query") or row.get("question")
        choices = _as_list(row.get("choices"))
        records.append(_record(sample_id, question, row.get("answer"), options=choices, image_path=image_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_mathverse_mini(skip_large_videos=False):
    parquet = hf_hub_download("AI4Math/MathVerse", "testmini.parquet", repo_type="dataset", local_dir=RAW_ROOT / "mathverse_mini")
    rows = pd.read_parquet(parquet).to_dict("records")
    out = DATA_ROOT / "mathverse_mini"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("sample_index") or f"mathverse_mini_{idx:06d}")
        image_path = _write_image(row.get("image"), out / "images", sample_id)
        question = row.get("query_wo") or row.get("question_for_eval") or row.get("question")
        options = _options_from_question(row.get("question_for_eval") or row.get("question") or "")
        records.append(_record(sample_id, question, row.get("answer"), options=options, image_path=image_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_videomme(skip_large_videos=False):
    rows = _load_parquets("lmms-lab/Video-MME", ["videomme/test-00000-of-00001.parquet"], "videomme")
    out = DATA_ROOT / "videomme"
    out.mkdir(parents=True, exist_ok=True)
    video_root = out / "videos"
    if not skip_large_videos:
        raw = snapshot_download(
            "lmms-lab/Video-MME",
            repo_type="dataset",
            local_dir=RAW_ROOT / "videomme",
            allow_patterns=["videos_chunked_*.zip", "subtitle.zip"],
        )
        _extract_zips(Path(raw).glob("videos_chunked_*.zip"), video_root)
    records = []
    for idx, row in enumerate(rows):
        video_id = str(row.get("video_id") or row.get("videoID") or f"{idx:03d}")
        video_path = _find_video(video_root, video_id, row.get("videoID"))
        records.append(_record(str(row.get("question_id") or f"videomme_{idx:06d}"), row.get("question"), row.get("answer"), options=_as_list(row.get("options")), video_path=video_path))
    _write_jsonl(out / "test.jsonl", records)
    _write_jsonl(out / "videomme.jsonl", records)


def prepare_videomme_short(skip_large_videos=False):
    rows = _load_parquets("lmms-lab/Video-MME", ["videomme/test-00000-of-00001.parquet"], "videomme")
    out = DATA_ROOT / "videomme_short"
    out.mkdir(parents=True, exist_ok=True)
    video_root = out / "videos"
    if not video_root.exists() and not skip_large_videos:
        prepare_videomme(skip_large_videos=False)
    records = []
    for idx, row in enumerate(rows):
        if str(row.get("duration") or "").lower() != "short":
            continue
        video_id = str(row.get("video_id") or row.get("videoID") or f"{idx:03d}")
        video_path = _find_video(video_root, video_id, row.get("videoID"))
        records.append(_record(str(row.get("question_id") or f"videomme_short_{idx:06d}"), row.get("question"), row.get("answer"), options=_as_list(row.get("options")), video_path=video_path))
    _write_jsonl(out / "videomme_short.jsonl", records)


def prepare_lvbench(skip_large_videos=False):
    rows = _load_parquets("lmms-lab/LVBench", ["data/train-00000-of-00001.parquet"], "lvbench")
    out = DATA_ROOT / "lvbench"
    out.mkdir(parents=True, exist_ok=True)
    video_root = out / "videos"
    if not skip_large_videos:
        raw = snapshot_download(
            "lmms-lab/LVBench",
            repo_type="dataset",
            local_dir=RAW_ROOT / "lvbench",
            allow_patterns=["video_chunks/*.zip"],
        )
        _extract_zips((Path(raw) / "video_chunks").glob("*.zip"), video_root)
    records = []
    for idx, row in enumerate(rows):
        q, options = _split_embedded_options(row.get("question") or "")
        video_path = _find_video(video_root, row.get("video_path"), row.get("key"))
        records.append(_record(str(row.get("uid") or f"lvbench_{idx:06d}"), q, row.get("answer"), options=options, video_path=video_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_longvideobench_val(skip_large_videos=False):
    out = DATA_ROOT / "longvideobench_val"
    out.mkdir(parents=True, exist_ok=True)
    raw = snapshot_download(
        "Jialuo21/LongVideoBench",
        repo_type="dataset",
        local_dir=RAW_ROOT / "longvideobench_val",
        allow_patterns=["lvb_val.json"] if skip_large_videos else ["lvb_val.json", "videos/*.mp4", "videos/*.json"],
    )
    data = json.loads((Path(raw) / "lvb_val.json").read_text(encoding="utf-8"))
    records = []
    for idx, row in enumerate(data):
        choices = _as_list(row.get("candidates"))
        answer_idx = row.get("correct_choice")
        answer = chr(ord("A") + int(answer_idx)) if answer_idx is not None else ""
        video_path = _rel_if_exists(out, Path(raw) / "videos" / str(row.get("video_path", "")))
        records.append(_record(str(row.get("id") or f"longvideobench_val_{idx:06d}"), row.get("question"), answer, options=choices, video_path=video_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_motionbench(skip_large_videos=False):
    out = DATA_ROOT / "motionbench"
    out.mkdir(parents=True, exist_ok=True)
    raw_dir = RAW_ROOT / "motionbench"
    meta_path = Path(hf_hub_download("zai-org/MotionBench", "MotionBench/video_info.meta.jsonl", repo_type="dataset", local_dir=raw_dir))
    if not skip_large_videos:
        snapshot_download(
            "zai-org/MotionBench",
            repo_type="dataset",
            local_dir=raw_dir,
            allow_patterns=[
                "MotionBench/public-dataset/*.mp4",
                "MotionBench/self-collected/*.mp4",
            ],
        )
    records = []
    with meta_path.open("r", encoding="utf-8") as handle:
        for idx, line in enumerate(handle):
            row = json.loads(line)
            video_path = "" if skip_large_videos else _find_existing_media_path(raw_dir / "MotionBench", row.get("video_path"))
            for qa_idx, qa in enumerate(row.get("qa") or []):
                answer = str(qa.get("answer") or "").strip()
                if not answer or answer == "NA":
                    continue
                q, options = _motion_question_options(qa)
                sample_id = qa.get("uid") or f"motionbench_{idx:06d}_{qa_idx:02d}"
                records.append(_record(str(sample_id), q, answer, options=options, video_path=video_path))
    _write_jsonl(out / "test.jsonl", records)


def prepare_pixmo_count(skip_large_videos=False):
    rows = _load_parquets("allenai/pixmo-count", ["data/test-00000-of-00001.parquet"], "pixmo_count")
    out = DATA_ROOT / "pixmo_count"
    (out / "images").mkdir(parents=True, exist_ok=True)
    image_paths = _download_url_images(rows, out / "images")
    records = []
    for idx, row in enumerate(rows):
        records.append(_record(
            f"pixmo_count_{idx:06d}",
            f"How many {row.get('label')} are in the image?",
            row.get("count"),
            image_path=image_paths.get(row.get("image_sha256"), ""),
            label=row.get("label"),
            image_url=row.get("image_url"),
        ))
    _write_jsonl(out / "test.jsonl", records)


def prepare_pixmo_pointing(skip_large_videos=False):
    rows = _load_parquets("allenai/pixmo-points-eval", ["data/test-00000-of-00001.parquet"], "pixmo_pointing")
    out = DATA_ROOT / "pixmo_pointing"
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    image_paths = _download_url_images(rows, out / "images")
    records = []
    for idx, row in enumerate(rows):
        points = _point_list(row.get("points"))
        mask_path = _write_point_masks(row.get("masks"), out / "masks", f"pixmo_pointing_{idx:06d}")
        records.append(_record(
            f"pixmo_pointing_{idx:06d}",
            f"Locate every {row.get('label')}.",
            _format_points(points),
            image_path=image_paths.get(row.get("image_sha256"), ""),
            label=row.get("label"),
            points=points,
            mask_path=mask_path,
            image_url=row.get("image_url"),
        ))
    _write_jsonl(out / "test.jsonl", records)


def prepare_refcoco(skip_large_videos=False):
    files = [f"data/val-{idx:05d}-of-00004.parquet" for idx in range(4)]
    rows = _load_parquets("lmms-lab/RefCOCO", files, "refcoco")
    out = DATA_ROOT / "refcoco"
    _reset_media_dirs(out, ["images"])
    records = []
    seen = set()
    for idx, row in enumerate(rows):
        sample_id = str(row.get("question_id") or f"refcoco_{idx:06d}")
        if sample_id in seen:
            continue
        seen.add(sample_id)
        image_path = _write_image(row.get("image"), out / "images", sample_id)
        answers = _as_list(row.get("answer"))
        bbox_xywh = [float(value) for value in row.get("bbox")]
        x, y, width, height = bbox_xywh
        bbox_xyxy = [x, y, x + width, y + height]
        with Image.open(out / image_path) as image:
            image_width, image_height = image.size
        bbox_normalized = [
            bbox_xyxy[0] / image_width,
            bbox_xyxy[1] / image_height,
            bbox_xyxy[2] / image_width,
            bbox_xyxy[3] / image_height,
        ]
        for expression_idx, expression in enumerate(answers or ["the referred object"]):
            records.append(_record(
                f"{sample_id}_{expression_idx}",
                expression,
                json.dumps(bbox_normalized),
                image_path=image_path,
                referring_expression=expression,
                bbox_xywh=bbox_xywh,
                bbox_xyxy=bbox_xyxy,
                bbox_normalized=bbox_normalized,
                image_width=image_width,
                image_height=image_height,
                segmentation=_json_value(row.get("segmentation")),
                file_name=row.get("file_name"),
            ))
    _write_jsonl(out / "test.jsonl", records)


def prepare_mmbench(skip_large_videos=False):
    rows = _load_parquets(
        "HuggingFaceM4/MMBench",
        ["data/validation-00000-of-00001-93861598ee8837c0.parquet"],
        "mmbench",
    )
    out = DATA_ROOT / "mmbench"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("index") or f"mmbench_{idx:06d}")
        image_path = _write_base64_image(row.get("image"), out / "images", sample_id)
        question = "\n\n".join(part for part in (_clean(row.get("hint")), _clean(row.get("question"))) if part)
        options = [_clean(row.get(key)) for key in ("A", "B", "C", "D") if _clean(row.get(key))]
        records.append(_record(
            sample_id,
            question,
            row.get("answer"),
            options=options,
            image_path=image_path,
            category=row.get("category"),
            l2_category=row.get("l2-category"),
        ))
    _write_jsonl(out / "test.jsonl", records)


def prepare_mme(skip_large_videos=False):
    files = [
        "data/test-00000-of-00004-a25dbe3b44c4fda6.parquet",
        "data/test-00001-of-00004-7d22c7f1aba6fca4.parquet",
        "data/test-00002-of-00004-594798fd3f5b029c.parquet",
        "data/test-00003-of-00004-53ae1794f93b1e35.parquet",
    ]
    rows = _load_parquets("lmms-lab/MME", files, "mme")
    out = DATA_ROOT / "mme"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        pair_id = str(row.get("question_id") or f"mme_pair_{idx:06d}")
        sample_id = f"mme_{idx:06d}"
        image_path = _write_image(row.get("image"), out / "images", pair_id)
        records.append(_record(
            sample_id,
            row.get("question"),
            row.get("answer"),
            image_path=image_path,
            category=row.get("category"),
            pair_id=pair_id,
        ))
    _write_jsonl(out / "test.jsonl", records)


def prepare_mmstar(skip_large_videos=False):
    rows = _load_parquets("Lin-Chen/MMStar", ["mmstar.parquet"], "mmstar")
    out = DATA_ROOT / "mmstar"
    _reset_media_dirs(out, ["images"])
    records = []
    for idx, row in enumerate(rows):
        sample_id = str(row.get("index") if row.get("index") is not None else f"mmstar_{idx:06d}")
        image_path = _write_image(row.get("image"), out / "images", sample_id)
        question, options = _split_mmstar_question(row.get("question"))
        records.append(_record(
            sample_id,
            question,
            row.get("answer"),
            options=options,
            image_path=image_path,
            category=row.get("category"),
            l2_category=row.get("l2_category"),
        ))
    _write_jsonl(out / "test.jsonl", records)


def prepare_mvbench(skip_large_videos=False):
    raw_dir = RAW_ROOT / "mvbench"
    raw = snapshot_download(
        "OpenGVLab/MVBench",
        repo_type="dataset",
        local_dir=raw_dir,
        allow_patterns=["json/*.json", "README.md", "video/*.txt"] if skip_large_videos else ["json/*.json", "README.md", "video/*"],
    )
    out = DATA_ROOT / "mvbench"
    out.mkdir(parents=True, exist_ok=True)
    video_root = out / "videos"
    if not skip_large_videos:
        _extract_zips((Path(raw) / "video").glob("*.zip"), video_root)
    media_index = _build_media_index(video_root)
    frame_dir_index = _build_frame_dir_index(video_root)
    records = []
    missing = []
    for annotation in sorted((Path(raw) / "json").glob("*.json")):
        task = annotation.stem
        for idx, row in enumerate(json.loads(annotation.read_text(encoding="utf-8"))):
            source_video = _find_video(video_root, row.get("video"), media_index=media_index)
            if not source_video:
                source_video = _frames_to_video(
                    frame_dir_index.get(Path(str(row.get("video"))).name),
                    video_root / "generated_from_frames" / f"{Path(str(row.get('video'))).name}.mp4",
                    video_root.parent,
                )
            video_path = source_video
            if source_video and row.get("start") is not None and row.get("end") is not None:
                if "generated_from_frames/" not in source_video or float(row.get("end")) <= _video_duration(out / source_video):
                    video_path = _clip_video(source_video, out / "clips", task, idx, row.get("start"), row.get("end"))
            if not video_path:
                missing.append({"task": task, "video": row.get("video")})
            options = _as_list(row.get("candidates"))
            answer = _answer_letter(row.get("answer"), options)
            records.append(_record(
                f"mvbench_{task}_{idx:04d}",
                row.get("question"),
                answer,
                options=options,
                video_path=video_path,
                task=task,
                source_answer=row.get("answer"),
                start=row.get("start"),
                end=row.get("end"),
            ))
    _write_jsonl(out / "test.jsonl", records)
    _write_jsonl(out / "missing_media.jsonl", missing)


def prepare_egoschema(skip_large_videos=False):
    rows = _load_parquets("lmms-lab/egoschema", ["Subset/test-00000-of-00001.parquet"], "egoschema")
    out = DATA_ROOT / "egoschema"
    out.mkdir(parents=True, exist_ok=True)
    video_root = out / "videos"
    video_ids = {str(row.get("video_idx")) for row in rows}
    if not skip_large_videos:
        raw = snapshot_download(
            "lmms-lab/egoschema",
            repo_type="dataset",
            local_dir=RAW_ROOT / "egoschema",
            allow_patterns=["videos_chunked_*.zip"],
        )
        _extract_selected_videos(Path(raw).glob("videos_chunked_*.zip"), video_root, video_ids)
    records = []
    for idx, row in enumerate(rows):
        video_id = str(row.get("video_idx"))
        options = _as_list(row.get("option"))
        answer_idx = int(row.get("answer"))
        records.append(_record(
            str(row.get("question_idx") or f"egoschema_{idx:05d}"),
            row.get("question"),
            chr(ord("A") + answer_idx),
            options=options,
            video_path=_find_video(video_root, video_id),
            video_id=video_id,
        ))
    _write_jsonl(out / "test.jsonl", records)


def _load_parquets(repo_id, files, raw_name):
    raw_dir = RAW_ROOT / raw_name
    paths = [hf_hub_download(repo_id, file, repo_type="dataset", local_dir=raw_dir) for file in files]
    frames = [pd.read_parquet(path) for path in paths]
    return pd.concat(frames, ignore_index=True).to_dict("records")


def _record(sample_id, question, answer, options=None, references=None, image_path="", video_path="", **metadata):
    record = {
        "id": str(sample_id),
        "question": "" if question is None else str(question).strip(),
        "options": options or [],
        "choices": options or [],
        "answer": "" if answer is None else str(answer).strip(),
        "references": references or ([answer] if answer not in (None, "") else []),
        "image_path": image_path or "",
        "video_path": video_path or "",
        "audio_path": "",
    }
    record.update({key: _json_value(value) for key, value in metadata.items()})
    return record


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"Wrote {len(records)} rows to {path}")


def _reset_media_dirs(root, names):
    root.mkdir(parents=True, exist_ok=True)
    for name in names:
        path = root / name
        if path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def _write_image(value, image_dir, stem):
    if value in (None, ""):
        return ""
    image_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(value, dict):
        data = value.get("bytes")
        source_path = value.get("path")
    elif isinstance(value, (bytes, bytearray, memoryview)):
        data = bytes(value)
        source_path = None
    else:
        data = None
        source_path = None
    suffix = Path(str(source_path or "")).suffix.lower() or ".jpg"
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        suffix = ".jpg"
    out = image_dir / f"{_safe_name(stem)}{suffix}"
    if data is not None:
        if isinstance(data, memoryview):
            data = data.tobytes()
        out.write_bytes(data)
        return str(out.relative_to(image_dir.parent))
    if source_path and Path(str(source_path)).exists():
        shutil.copy2(source_path, out)
        return str(out.relative_to(image_dir.parent))
    return ""


def _write_base64_image(value, image_dir, stem):
    if value in (None, ""):
        return ""
    try:
        data = base64.b64decode(str(value), validate=True)
    except (ValueError, TypeError):
        return ""
    return _write_image(data, image_dir, stem)


def _download_url_images(rows, image_dir):
    image_dir.mkdir(parents=True, exist_ok=True)
    jobs = {}
    for row in rows:
        digest = str(row.get("image_sha256") or "")
        url = str(row.get("image_url") or "")
        if digest and url:
            jobs[digest] = url

    paths = {}
    failures = []
    with ThreadPoolExecutor(max_workers=16) as pool:
        futures = {pool.submit(_download_one_image, url, digest, image_dir): digest for digest, url in jobs.items()}
        for future in as_completed(futures):
            digest = futures[future]
            try:
                paths[digest] = future.result()
            except Exception as exc:
                failures.append({"image_sha256": digest, "image_url": jobs[digest], "error": str(exc)})
    if failures:
        _write_jsonl(image_dir.parent / "missing_images.jsonl", failures)
    else:
        (image_dir.parent / "missing_images.jsonl").unlink(missing_ok=True)
    print(f"Downloaded {len(paths)}/{len(jobs)} URL images to {image_dir}")
    return paths


def _download_one_image(url, expected_sha256, image_dir):
    existing = list(image_dir.glob(f"{expected_sha256}.*"))
    if existing:
        return str(existing[0].relative_to(image_dir.parent))
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=45) as response:
        data = response.read()
        content_type = response.headers.get_content_type()
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected_sha256:
        raise ValueError(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")
    suffix = {
        "image/png": ".png",
        "image/webp": ".webp",
        "image/bmp": ".bmp",
    }.get(content_type, ".jpg")
    output = image_dir / f"{expected_sha256}{suffix}"
    output.write_bytes(data)
    return str(output.relative_to(image_dir.parent))


def _extract_zips(paths, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    for path in sorted(paths):
        marker = output_dir / f".{path.name}.done"
        if marker.exists():
            continue
        print(f"Extracting {path}...")
        with zipfile.ZipFile(path) as archive:
            archive.extractall(output_dir)
        marker.write_text("ok\n", encoding="utf-8")


def _extract_selected_videos(paths, output_dir, video_ids):
    output_dir.mkdir(parents=True, exist_ok=True)
    remaining = {video_id for video_id in video_ids if not list(output_dir.glob(f"{video_id}.*"))}
    for path in sorted(paths):
        if not remaining:
            break
        print(f"Scanning {path.name} for {len(remaining)} requested EgoSchema videos...")
        with zipfile.ZipFile(path) as archive:
            selected = [
                name for name in archive.namelist()
                if Path(name).stem in remaining and not name.endswith("/")
            ]
            for name in selected:
                target = output_dir / Path(name).name
                with archive.open(name) as source, target.open("wb") as destination:
                    shutil.copyfileobj(source, destination)
                remaining.discard(Path(name).stem)
    if remaining:
        print(f"Warning: {len(remaining)} EgoSchema subset videos were not found in the archives.")


def _find_video(video_root, *names, media_index=None):
    if not video_root.exists():
        return ""
    candidates = []
    for name in names:
        if name in (None, ""):
            continue
        p = Path(str(name))
        candidates.append(video_root / p)
        candidates.append(video_root / p.name)
        if not p.suffix:
            candidates.append(video_root / f"{p}.mp4")
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.relative_to(video_root.parent))
    for name in names:
        if name in (None, ""):
            continue
        key = Path(str(name)).name
        if media_index is None:
            matches = list(video_root.rglob(key))
            if not matches and not Path(str(name)).suffix:
                matches = list(video_root.rglob(f"{key}.mp4"))
        else:
            matches = media_index.get(key, [])
            if not matches and not Path(str(name)).suffix:
                matches = media_index.get(f"{key}.mp4", []) or media_index.get(Path(key).stem, [])
        matches = [match for match in matches if match.is_file()]
        if matches:
            return str(matches[0].relative_to(video_root.parent))
    return ""


def _build_media_index(root):
    index = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if not path.is_file() or path.name.startswith("."):
            continue
        index.setdefault(path.name, []).append(path)
        index.setdefault(path.stem, []).append(path)
    return index


def _build_frame_dir_index(root):
    index = {}
    if not root.exists():
        return index
    for path in root.rglob("*"):
        if path.is_dir():
            index.setdefault(path.name, path)
    return index


def _frames_to_video(frame_dir, output, dataset_root):
    if frame_dir is None:
        return ""
    if output.exists():
        return str(output.relative_to(dataset_root))
    first_frame = next(iter(sorted(frame_dir.glob("*.jpg"))), None)
    if first_frame is None:
        return ""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-framerate", "3",
        "-i", str(frame_dir / "%05d.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p", str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        output.unlink(missing_ok=True)
        return ""
    return str(output.relative_to(dataset_root))


def _clip_video(source_video, clip_root, task, idx, start, end):
    source = clip_root.parent / source_video
    if not source.is_file():
        return source_video
    output = clip_root / task / f"{idx:04d}.mp4"
    if output.exists():
        return str(output.relative_to(clip_root.parent))
    output.parent.mkdir(parents=True, exist_ok=True)
    duration = max(0.0, float(end) - float(start))
    command = [
        "ffmpeg", "-loglevel", "error", "-y", "-ss", str(start), "-i", str(source),
        "-t", str(duration), "-map", "0:v:0", "-map", "0:a?",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-c:a", "aac", str(output),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True)
    except (OSError, subprocess.CalledProcessError):
        output.unlink(missing_ok=True)
        return source_video
    return str(output.relative_to(clip_root.parent))


def _video_duration(path):
    try:
        output = subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
            text=True,
        )
        return float(output.strip())
    except (OSError, subprocess.CalledProcessError, ValueError):
        return 0.0


def _find_existing_media_path(root, value):
    if value in (None, ""):
        return ""
    direct = root / str(value)
    if direct.exists():
        return str(direct)
    matches = list(root.rglob(Path(str(value)).name))
    return str(matches[0]) if matches else ""


def _rel_if_exists(root, path):
    if not path.exists():
        return ""
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _as_list(value):
    if hasattr(value, "tolist"):
        return _as_list(value.tolist())
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()]


def _options_from_question(question):
    pairs = re.findall(r"([A-I])\s*[:\.\)]\s*([^A-I\n]+?)(?=(?:\s+[A-I]\s*[:\.\)]|$))", str(question))
    return [text.strip() for _, text in pairs]


def _split_embedded_options(question):
    text = str(question).strip()
    matches = list(re.finditer(r"\(([A-I])\)\s*", text))
    if not matches:
        return text, []
    q = text[: matches[0].start()].strip()
    options = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        options.append(text[start:end].strip())
    return q, options


def _motion_question_options(row):
    question = row.get("question") or row.get("Question") or ""
    options = row.get("options") or row.get("choices") or []
    if not options:
        options = [row.get(key) for key in ("A", "B", "C", "D") if row.get(key) not in (None, "")]
    if not options:
        question, options = _split_embedded_options(question)
    return question, _as_list(options)


def _safe_name(value):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value))[:160]


def _clean(value):
    return "" if value is None else str(value).strip()


def _json_value(value):
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def _point_list(value):
    points = _json_value(value) or []
    if isinstance(points, dict):
        xs = points.get("x", [])
        ys = points.get("y", [])
        return [{"x": float(x), "y": float(y)} for x, y in zip(xs, ys)]
    return [
        {"x": float(point["x"]), "y": float(point["y"])}
        for point in points
        if isinstance(point, dict) and "x" in point and "y" in point
    ]


def _format_points(points):
    return "; ".join(f"({point['x']:.2f}, {point['y']:.2f})" for point in points)


def _write_point_masks(value, mask_dir, stem):
    masks = [] if value is None else list(value)
    if not masks:
        return ""
    output = mask_dir / f"{stem}.npz"
    arrays = {
        f"mask_{idx}": np.stack([np.asarray(row, dtype=bool) for row in mask])
        for idx, mask in enumerate(masks)
    }
    np.savez_compressed(output, **arrays)
    return str(output.relative_to(mask_dir.parent))


def _split_mmstar_question(value):
    text = _clean(value)
    marker = re.search(r"\n(?:Options|Choices):\s*", text, re.IGNORECASE)
    if not marker:
        marker = re.search(r"\n(?=\([A-I]\)\s*)", text)
    if not marker:
        return text, []
    question = text[:marker.start()].strip()
    question_match = re.search(r"(?:^|\n)Question:\s*(.*)", question, re.IGNORECASE | re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()
    option_text = text[marker.end():]
    if re.search(r"(?:^|\n)\([A-I]\)\s*", option_text):
        matches = list(re.finditer(r"(?:^|\n)\(([A-I])\)\s*", option_text))
    else:
        matches = list(re.finditer(r"(?:^|,\s+)([A-I]):\s*", option_text))
    options = []
    for idx, match in enumerate(matches):
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(option_text)
        options.append(option_text[start:end].strip().rstrip(","))
    return question, options


def _answer_letter(answer, options):
    answer = _clean(answer)
    for idx, option in enumerate(options):
        if _clean(option).lower() == answer.lower():
            return chr(ord("A") + idx)
    if re.fullmatch(r"[A-I]", answer, re.IGNORECASE):
        return answer.upper()
    return answer


PREPARE = {
    "chartqa": prepare_chartqa,
    "ocrbench": prepare_ocrbench,
    "textvqa": prepare_textvqa,
    "docvqa": prepare_docvqa,
    "infographicvqa": prepare_infographicvqa,
    "mathvista": prepare_mathvista,
    "mathverse_mini": prepare_mathverse_mini,
    "videomme": prepare_videomme,
    "videomme_short": prepare_videomme_short,
    "lvbench": prepare_lvbench,
    "longvideobench_val": prepare_longvideobench_val,
    "motionbench": prepare_motionbench,
    "pixmo_count": prepare_pixmo_count,
    "pixmo_pointing": prepare_pixmo_pointing,
    "refcoco": prepare_refcoco,
    "mmbench": prepare_mmbench,
    "mme": prepare_mme,
    "mmstar": prepare_mmstar,
    "mvbench": prepare_mvbench,
    "egoschema": prepare_egoschema,
}


if __name__ == "__main__":
    main()
