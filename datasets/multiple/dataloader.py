from datasets._code_common import load_multiple_dataset


eval_type = "open"


def load_data(data_dir=None):
    return load_multiple_dataset(data_dir)


load_dataset = load_data
