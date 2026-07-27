from datasets._math_common import (
    SYSTEM_PROMPT_MATH,
    build_symbolic_math_prompt,
    postprocess_symbolic_math_prediction,
)


SYSTEM_PROMPT = SYSTEM_PROMPT_MATH
build_prompt = build_symbolic_math_prompt
postprocess_prediction = postprocess_symbolic_math_prediction
