def lcs_length(a: list[str], b: list[str]) -> int:
    # 短い方を内側ループにして空間 O(min(n, m))
    if len(a) < len(b):
        a, b = b, a
    prev = [0] * (len(b) + 1)
    for x in a:
        cur = [0] * (len(b) + 1)
        for j, y in enumerate(b, 1):
            if x == y:
                cur[j] = prev[j - 1] + 1
            else:
                cur[j] = cur[j - 1] if cur[j - 1] > prev[j] else prev[j]
        prev = cur
    return prev[-1]


def rouge_l(candidate: list[str], reference: list[str], beta: float = 1.0) -> tuple[float, float, float]:
    if not candidate or not reference:
        return (0.0, 0.0, 0.0)
    lcs = lcs_length(candidate, reference)
    if lcs == 0:
        return (0.0, 0.0, 0.0)
    p = lcs / len(candidate)
    r = lcs / len(reference)
    f = (1 + beta**2) * p * r / (r + beta**2 * p)
    return (p, r, f)
