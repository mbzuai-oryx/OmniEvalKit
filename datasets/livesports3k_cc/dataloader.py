from datasets._audio_common import load_jsonl_dataset


eval_type = "open"
DATASET_NAME = "livesports3k_cc"


def load_data(data_dir=None):
    return load_jsonl_dataset(DATASET_NAME, data_dir)


load_dataset = load_data
