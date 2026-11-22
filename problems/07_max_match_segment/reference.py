from typing import Iterable


def _prepare(dictionary: Iterable[str]) -> tuple[set[str], int]:
    words = {w for w in dictionary if w}
    return words, max(map(len, words), default=0)


def segment(text: str, dictionary: Iterable[str]) -> list[str]:
    words, max_len = _prepare(dictionary)
    result, i, n = [], 0, len(text)
    while i < n:
        for length in range(min(max_len, n - i), 0, -1):
            if text[i:i + length] in words:
                break
        else:
            length = 1  # 辞書に無ければ1文字
        result.append(text[i:i + length])
        i += length
    return result


def segment_backward(text: str, dictionary: Iterable[str]) -> list[str]:
    words, max_len = _prepare(dictionary)
    result, j = [], len(text)
    while j > 0:
        for length in range(min(max_len, j), 0, -1):
            if text[j - length:j] in words:
                break
        else:
            length = 1
        result.append(text[j - length:j])
        j -= length
    result.reverse()
    return result
