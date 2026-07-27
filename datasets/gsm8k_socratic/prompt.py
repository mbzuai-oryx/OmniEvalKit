from datasets._math_common import SYSTEM_PROMPT_MATH, build_gsm8k_prompt, postprocess_gsm8k_prediction


SYSTEM_PROMPT = SYSTEM_PROMPT_MATH
build_prompt = build_gsm8k_prompt
postprocess_prediction = postprocess_gsm8k_prediction
