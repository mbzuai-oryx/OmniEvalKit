from datasets._audio_common import SYSTEM_PROMPT_AST


SYSTEM_PROMPT = SYSTEM_PROMPT_AST


def build_prompt(question=None, options=None):
    return "Translate the English audio transcript into Chinese. Output only the translation."
