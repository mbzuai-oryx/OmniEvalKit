from datasets._vision_video_common import load_jsonl_dataset
from datasets._vision_video_common import compute_mcq_score as compute_score


eval_type = "closed"


def load_data(data_dir=None):
    return load_jsonl_dataset("egoschema", data_dir)


load_dataset = load_data
