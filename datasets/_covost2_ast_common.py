from pathlib import Path

from datasets._audio_common import SYSTEM_PROMPT_AST, build_ast_prompt, load_jsonl_dataset


eval_type = "open"
SYSTEM_PROMPT = SYSTEM_PROMPT_AST
build_prompt = build_ast_prompt


def load_covost2_dataset(dataset_name, data_dir=None):
    root = Path(data_dir) if data_dir else Path(__file__).resolve().parents[1] / "data" / dataset_name
    manifest = root / "test.jsonl"
    if not manifest.exists():
        lang = dataset_name.removeprefix("covost2_").removesuffix("_en").replace("_", "-")
        if lang == "zh":
            lang = "zh-CN"
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest}. This is an evaluation-only CoVoST2 loader, "
            f"but the official facebook/covost2 dataset does not bundle audio. Prepare the test split with:\n"
            f"  python scripts/prepare_covost2_x_en.py {lang} --split test\n"
            f"It requires Common Voice Corpus 4 audio at data/commonvoice_corpus4/{lang}/validated.tsv "
            f"and data/commonvoice_corpus4/{lang}/clips/*.mp3."
        )
    return load_jsonl_dataset(dataset_name, data_dir)
