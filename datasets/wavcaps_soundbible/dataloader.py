from datasets._audio_common import load_jsonl_dataset


eval_type = "open"
DATASET_NAME = "wavcaps_soundbible"


def load_data(data_dir=None):
    return load_jsonl_dataset(DATASET_NAME, data_dir)


load_dataset = load_data
