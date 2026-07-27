from datasets._math_common import load_math_jsonl_dataset


eval_type = "open"
DATASET_NAME = "gsm8k_socratic"


def load_data(data_dir=None):
    return load_math_jsonl_dataset(DATASET_NAME, data_dir)


load_dataset = load_data
