import os, sys
import argparse
import string

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from pypinyin import lazy_pinyin
from tn.chinese.normalizer import Normalizer as Normalizer_zh
from tn.english.normalizer import Normalizer as Normalizer_en
from zhon.hanzi import punctuation
from cn_tn import TextNorm


def process_text(text: str, lang: str) -> str:
    if not text.strip():
        raise ValueError("Input text cannot be empty")

    if lang == "zh":
        try:
            normalizer = Normalizer_zh()
            tn_text = normalizer.normalize(text)
            py = lazy_pinyin(tn_text)
            return py
        except Exception as e:
            raise ValueError(f"Failed to normalize Chinese text: {e}")
    elif lang == "en":
        try:
            normalizer = Normalizer_en()
            tn_text = normalizer.normalize(text)
            return tn_text
        except Exception as e:
            raise ValueError(f"Failed to normalize English text: {e}")
    else:
        raise ValueError(
            "Unsupported language. Use 'zh' for Chinese or 'en' for English"
        )


def zh_norm(text):
    normalizer = TextNorm()
    return normalizer(text)


def zh_remove_punc(text):
    punctuation_all = punctuation + string.punctuation
    for x in punctuation_all:
        text = text.replace(x, "")

    text = text.replace("  ", " ")
    return text


def en_remove_punc(text):
    punctuation_all = punctuation + string.punctuation
    punctuation_all = [i for i in punctuation_all if (i != "'" and i != "-")]

    for x in punctuation_all:
        text = text.replace(x, "")

    text = text.replace("  ", " ")
    return text
