#!/usr/bin/env python3
import argparse
import csv
import json
import re
import shutil
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd


XX_EN_LANGUAGES = {
    "fr": "French",
    "de": "German",
    "es": "Spanish",
    "ca": "Catalan",
    "it": "Italian",
    "ru": "Russian",
    "zh-CN": "Chinese",
    "pt": "Portuguese",
    "fa": "Persian",
    "et": "Estonian",
    "mn": "Mongolian",
    "nl": "Dutch",
    "tr": "Turkish",
    "ar": "Arabic",
    "sv-SE": "Swedish",
    "lv": "Latvian",
    "sl": "Slovenian",
    "ta": "Tamil",
    "ja": "Japanese",
    "id": "Indonesian",
    "cy": "Welsh",
}

TEST_SPLIT_SIZES = {
    "fr": 14760,
}

COVOST_URL_TEMPLATE = "https://dl.fbaipublicfiles.com/covost/covost_v2.{src_lang}_en.tsv.tar.gz"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("languages", nargs="*", help="CoVoST2 source language codes, or 'all'.")
    parser.add_argument("--data-root", default="data")
    parser.add_argument("--raw-root", default="data/raw_covost2")
    parser.add_argument("--commonvoice-root", default="data/commonvoice_corpus4")
    parser.add_argument("--split", default="test", choices=["train", "dev", "test"])
    parser.add_argument("--copy-audio", action="store_true", help="Copy audio files instead of symlinking them.")
    parser.add_argument("--download-timeout", type=int, default=60)
    parser.add_argument("--allow-partial", action="store_true", help="Write available rows even when local audio coverage is incomplete.")
    args = parser.parse_args()

    languages = list(XX_EN_LANGUAGES) if not args.languages or "all" in args.languages else args.languages
    unknown = [lang for lang in languages if lang not in XX_EN_LANGUAGES]
    if unknown:
        raise ValueError(f"Unknown CoVoST2 X->en language code(s): {', '.join(unknown)}")

    for lang in languages:
        prepare_language(
            lang,
            Path(args.data_root),
            Path(args.raw_root),
            Path(args.commonvoice_root),
            args.split,
            copy_audio=args.copy_audio,
            download_timeout=args.download_timeout,
            allow_partial=args.allow_partial,
        )


def prepare_language(lang, data_root, raw_root, commonvoice_root, split, copy_audio=False, download_timeout=60, allow_partial=False):
    dataset_name = dataset_name_for(lang)
    cv_root = find_commonvoice_root(commonvoice_root, lang)
    prepared_cv_manifest = data_root / f"commonvoice_{dataset_suffix(lang)}" / "test.jsonl"
    if not cv_root and not prepared_cv_manifest.exists():
        raise FileNotFoundError(
            f"Common Voice Corpus 4 directory not found for {lang}. Expected one of: "
            f"{commonvoice_root / lang}, {commonvoice_root / lang.replace('-', '_')}, "
            f"{commonvoice_root / lang.split('-')[0]}. "
            f"As a fallback, this script can also use {prepared_cv_manifest} if it contains original "
            "Common Voice path metadata."
        )

    covost_tsv = download_covost_tsv(lang, raw_root, timeout=download_timeout)
    covost_df = read_tsv(covost_tsv)
    if cv_root:
        validated_tsv = cv_root / "validated.tsv"
        clips_dir = cv_root / "clips"
        if not validated_tsv.exists() or not clips_dir.exists():
            raise FileNotFoundError(f"Expected {validated_tsv} and {clips_dir}")
        cv_df = read_tsv(validated_tsv)
        merged = pd.merge(
            left=cv_df[["path", "sentence", "client_id"]],
            right=covost_df[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        audio_lookup = {row["path"]: clips_dir / row["path"] for row in merged.to_dict("records")}
    else:
        cv_rows = load_prepared_commonvoice_rows(prepared_cv_manifest)
        cv_df = pd.DataFrame(
            {
                "path": path,
                "sentence": row.get("sentence", ""),
                "client_id": row.get("client_id", ""),
            }
            for path, row in cv_rows.items()
        )
        merged = pd.merge(
            left=cv_df[["path", "sentence", "client_id"]],
            right=covost_df[["path", "translation", "split"]],
            how="inner",
            on="path",
        )
        audio_lookup = {path: Path(row["audio_path"]) for path, row in cv_rows.items()}
    if split == "train":
        merged = merged[(merged["split"] == "train") | (merged["split"] == "train_covost")]
    else:
        merged = merged[merged["split"] == split]

    out_dir = data_root / dataset_name
    audio_dir = out_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "test.jsonl"
    source_name = XX_EN_LANGUAGES[lang]

    written = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        for idx, row in enumerate(merged.to_dict("records")):
            source_audio = audio_lookup.get(row["path"])
            if source_audio is None or not source_audio.exists():
                continue
            target_audio = audio_dir / f"{dataset_name}_{idx:06d}{source_audio.suffix.lower() or '.mp3'}"
            materialize_audio(source_audio, target_audio, copy_audio=copy_audio)
            record_id = f"{dataset_name}_{idx:06d}"
            question = f"Translate the spoken {source_name} audio into English."
            answer = str(row["translation"]).strip()
            record = {
                "id": record_id,
                "name": record_id,
                "question": question,
                "prompt": question,
                "options": [],
                "choices": [],
                "answer": answer,
                "reference": answer,
                "WavPath": str(target_audio.relative_to(out_dir)),
                "audio_path": str(target_audio.relative_to(out_dir)),
                "VideoPath": "",
                "video_path": "",
                "image_path": "",
                "source_sentence": str(row["sentence"]).strip(),
                "text": answer,
                "split": split,
                "source_language": lang,
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    expected = TEST_SPLIT_SIZES.get(lang) if split == "test" else None
    if expected is not None and written != expected and not allow_partial:
        manifest_path.unlink(missing_ok=True)
        if audio_dir.exists():
            shutil.rmtree(audio_dir)
        raise RuntimeError(
            f"[{dataset_name}] incomplete local audio coverage: wrote {written}/{expected} test samples. "
            f"Deleted partial output. Provide complete Common Voice 4 {lang} audio or rerun with --allow-partial."
        )

    print(f"[{dataset_name}] wrote {written} samples -> {manifest_path}", flush=True)


def download_covost_tsv(lang, raw_root, timeout=60):
    raw_root.mkdir(parents=True, exist_ok=True)
    tsv_path = raw_root / f"covost_v2.{lang}_en.tsv"
    if tsv_path.exists():
        return tsv_path

    archive_path = raw_root / f"covost_v2.{lang}_en.tsv.tar.gz"
    if not archive_path.exists():
        url = COVOST_URL_TEMPLATE.format(src_lang=lang)
        print(f"Downloading {url}", flush=True)
        with urllib.request.urlopen(url, timeout=timeout) as response:
            archive_path.write_bytes(response.read())

    with tarfile.open(archive_path, "r:gz") as archive:
        member = next((item for item in archive.getmembers() if item.name.endswith(".tsv")), None)
        if member is None:
            raise RuntimeError(f"No TSV found in {archive_path}")
        member.name = Path(member.name).name
        archive.extract(member, raw_root)
    return tsv_path


def find_commonvoice_root(commonvoice_root, lang):
    candidates = [
        commonvoice_root / lang,
        commonvoice_root / lang.replace("-", "_"),
        commonvoice_root / lang.split("-")[0],
    ]
    for candidate in candidates:
        if (candidate / "validated.tsv").exists() and (candidate / "clips").exists():
            return candidate
    return None


def load_prepared_commonvoice_rows(manifest_path):
    rows = {}
    root = manifest_path.parent
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            path = row.get("path")
            audio_path = row.get("audio_path")
            if not path or not audio_path:
                continue
            resolved_audio_path = Path(audio_path)
            if not resolved_audio_path.is_absolute():
                resolved_audio_path = root / resolved_audio_path
            row = dict(row)
            row["audio_path"] = str(resolved_audio_path.resolve())
            rows[path] = row
    return rows


def read_tsv(path):
    return pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )


def materialize_audio(source, target, copy_audio=False):
    if target.exists():
        return
    if copy_audio:
        shutil.copy2(source, target)
        return
    try:
        target.symlink_to(source.resolve())
    except OSError:
        shutil.copy2(source, target)


def dataset_name_for(lang):
    normalized = dataset_suffix(lang)
    return f"covost2_{re.sub(r'[^a-z0-9_]+', '_', normalized)}_en"


def dataset_suffix(lang):
    normalized = lang.lower().replace("-", "_")
    if normalized == "zh_cn":
        normalized = "zh"
    return normalized


if __name__ == "__main__":
    main()
