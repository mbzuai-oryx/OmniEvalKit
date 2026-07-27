from datasets._vision_video_common import load_jsonl_dataset
from datasets._vision_video_common import compute_mcq_score as compute_score


eval_type = "closed"


def load_data(data_dir=None):
    return [sample for sample in load_jsonl_dataset("mvbench", data_dir) if sample.get("video_path")]


load_dataset = load_data
