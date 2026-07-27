from datasets._vision_video_common import load_jsonl_dataset
from datasets._vision_video_common import compute_mcq_score as compute_score
from utils.evaluate import aggregate_scores as _aggregate_scores


eval_type = "closed"


def load_data(data_dir=None):
    return load_jsonl_dataset("mmbench", data_dir)


load_dataset = load_data


def aggregate_scores(results, eval_type="closed"):
    summary = _aggregate_scores(results, eval_type)
    summary["evaluation_protocol"] = "VanillaEval"
    return summary
