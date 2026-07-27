from datasets._covost2_ast_common import load_covost2_dataset


eval_type = "open"
DATASET_NAME = "covost2_ar_en"


def load_data(data_dir=None):
    return load_covost2_dataset(DATASET_NAME, data_dir)


load_dataset = load_data

