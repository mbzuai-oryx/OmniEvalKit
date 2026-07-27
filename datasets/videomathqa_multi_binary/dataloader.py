from datasets._videomathqa_common import load_videomathqa_split


eval_type = "closed"
DATASET_NAME = "videomathqa_multi_binary"


def load_data(data_dir=None):
    return load_videomathqa_split("videomathqa_mbin_test.parquet", "mbinary_id", data_dir=data_dir)


load_dataset = load_data
