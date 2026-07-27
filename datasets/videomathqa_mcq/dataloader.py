from datasets._videomathqa_common import load_videomathqa_split


eval_type = "closed"
DATASET_NAME = "videomathqa_mcq"


def load_data(data_dir=None):
    return load_videomathqa_split("videomathqa_mcq_test.parquet", "question_id", data_dir=data_dir)


load_dataset = load_data
