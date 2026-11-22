def levenshtein(a: str, b: str) -> int:
    if len(a) < len(b):
        a, b = b, a  # b を短い方にしてメモリ O(min(|a|, |b|))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1,               # 削除
                         cur[j - 1] + 1,            # 挿入
                         prev[j - 1] + (ca != cb))  # 置換 (一致ならコスト0)
        prev = cur
    return prev[-1]


def suggest(word: str, vocab: list[str], k: int, max_distance: int | None = None) -> list[tuple[str, int]]:
    if k <= 0:
        return []
    scored = []
    for w in set(vocab):
        # 長さの差は距離の下界なので、計算せずに枝刈りできる
        if max_distance is not None and abs(len(w) - len(word)) > max_distance:
            continue
        d = levenshtein(word, w)
        if max_distance is None or d <= max_distance:
            scored.append((w, d))
    scored.sort(key=lambda x: (x[1], x[0]))
    return scored[:k]
