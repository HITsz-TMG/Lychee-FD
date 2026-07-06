def check_backchannel(s: str) -> bool:
    """
    The model itself has backchannel judgment capabilities; this is a fallback function.
    It determines if the text is empty, meaningless, or belongs to Chinese/English backchannels.
    """

    s = s.strip().replace(",", "").replace(".", "").replace("，", "").replace("。", "")

    # Completely empty
    if not s:
        return True

    # === Chinese and English backchannels ===
    BACKCHANNEL = {
        # Chinese
        "嗯",
        "嗯嗯",
        "啊",
        "啊啊",
        "哦",
        "哦哦",
        "噢",
        "噢噢",
        "哎",
        "好",
        "好的",
        "好啊",
        "好吧",
        "好嘞",
        "对",
        "对对",
        "是",
        "是的",
        "行",
        "可以",
        "嗯哼",
        "哼",
        "嘿",
        # English
        "ok",
        "okay",
        "yeah",
        "yep",
        "yup",
        "right",
        "sure",
        "uh-huh",
        "uh huh",
        "uhhuh",
        "hmm",
        "hmmm",
        "mm",
        "mmm",
        "alright",
        "got it",
        "i see",
        "roger",
        "k",
        "kk",
        "y",
        "yep",
        "yes",
    }

    # Unify to lowercase for English judgement
    if s.lower() in BACKCHANNEL:
        return True

    # Some backchannels might be recognized as long words in ASR, handling short phrase patterns here
    if len(s) <= 5 and any(
        key in s.lower() for key in ["ok", "mm", "hmm", "uh", "yes", "yeah"]
    ):
        return True
    if len(s) <= 2 and any(ch in s for ch in ["嗯", "啊", "哦"]):
        return True

    # Pure punctuation
    if all(ch in ".,!?;。，！？；…" for ch in s):
        return True

    return False


def remove_leading_backchannel(text: str) -> str:
    """
    Scans the string from the beginning, removing leading backchannel and punctuation,
    until the first non-backchannel character is encountered.
    A fallback function to prevent edge cases when the model loops infinitely.
    """
    # Common Chinese backchannel characters
    backchannel_chars = {"嗯", "啊", "哦", "噢", "呃", "哎", "哼", "嘿"}
    # Common punctuation and whitespace
    punctuation_chars = {
        " ",
        ",",
        ".",
        "?",
        "!",
        "，",
        "。",
        "？",
        "！",
        "、",
        "；",
        ";",
        "…",
        ":",
        "：",
    }

    skip_chars = backchannel_chars.union(punctuation_chars)

    for i, char in enumerate(text):
        if char not in skip_chars:
            return text[i:]

    return ""
