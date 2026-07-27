from datasets._audio_common import SYSTEM_PROMPT_AST


SYSTEM_PROMPT = SYSTEM_PROMPT_AST


def build_prompt(question=None, options=None):
    return "Translate the Chinese audio transcript into English. Output only the translation."
