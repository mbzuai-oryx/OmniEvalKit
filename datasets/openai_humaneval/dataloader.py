from datasets._code_common import load_code_jsonl_dataset


eval_type = "open"
DATASET_NAME = "openai_humaneval"


def load_data(data_dir=None):
    return load_code_jsonl_dataset(DATASET_NAME, data_dir)


load_dataset = load_data
