import re


def split_cn_en(text: str):
    """
    Split text into a list containing single Chinese characters, English words, or numbers.
    """
    pattern = r"[\u4e00-\u9fff]|[A-Za-z]+|[0-9]+"
    return re.findall(pattern, text)


def check_en(text: str):
    """
    Check if the text is primarily English.
    Iterates backwards, ignoring symbols and digits. If a Chinese character is found, returns False.
    """
    if not text:
        return False

    symbol_pattern = re.compile(
        r"[\u0020-\u002F\u003A-\u0040\u005B-\u0060\u007B-\u007E"
        r"\u2000-\u206F"
        r"\u3000-\u303F"
        r"\uFF00-\uFFEF]"
    )

    for char in reversed(text):
        if char.isdigit() or symbol_pattern.match(char):
            continue
        if char >= "\u4e00" and char <= "\u9fff":  # is chinese
            return False
        else:
            return True

    return True


def detect_language_accent(text: str) -> str:
    """
    Determine whether the text should be spoken with a Chinese ('zh') or English ('en') accent.
    Logic:
    1. If no Chinese characters are present, default to 'en'.
    2. If Chinese characters are present, compare the count of Chinese characters vs English words.
    3. If Chinese characters count >= English words count, prefer 'zh' (Chinese engines usually handle English words okay).
    4. Otherwise, if English dominates significantly, use 'en'.
    """
    if not text:
        return "zh"

    # Count Chinese characters
    zh_chars = re.findall(r"[\u4e00-\u9fff]", text)
    num_zh = len(zh_chars)

    # Count English words
    en_words = re.findall(r"[a-zA-Z]+", text)
    num_en = len(en_words)

    if num_zh == 0:
        return "en"

    if num_zh >= num_en:
        return "zh"

    return "en"


def get_lcs_substrings(s1: str, s2: str) -> tuple[str, str]:
    """
    Calculate the Longest Common Subsequence (LCS) of two strings.
    Return the substrings of both strings starting from the first character of the LCS.

    Args:
        s1 (str): First string.
        s2 (str): Second string.

    Returns:
        tuple[str, str]: Substrings of s1 and s2 starting from the first char of the LCS.
                         If no common subsequence is found, returns original strings.
    """
    if not s1 or not s2:
        return s1, s2

    m, n = len(s1), len(s2)
    # dp[i][j] stores the length of LCS of s1[i:] and s2[j:]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            if s1[i] == s2[j]:
                dp[i][j] = 1 + dp[i + 1][j + 1]
            else:
                dp[i][j] = max(dp[i + 1][j], dp[i][j + 1])

    max_len = dp[0][0]
    if max_len == 0:
        return s1, s2

    # Find the start indices of the LCS
    start_i, start_j = -1, -1
    for i in range(m):
        for j in range(n):
            # If characters match and this match contributes to the max length LCS starting from here
            if s1[i] == s2[j] and dp[i][j] == max_len:
                start_i, start_j = i, j
                break
        if start_i != -1:
            break

    return s1[start_i:], s2[start_j:]
