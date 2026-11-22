OPEN = set("「『（(")
CLOSE = set("」』）)")
TERMINALS = set("。！？!?")


def split_sentences(text: str) -> list[str]:
    sentences: list[str] = []

    def flush(buf: list[str]) -> None:
        s = "".join(buf).strip()
        if s:
            sentences.append(s)

    for line in text.split("\n"):
        depth = 0
        buf: list[str] = []
        for i, ch in enumerate(line):
            buf.append(ch)
            if ch in OPEN:
                depth += 1
            elif ch in CLOSE:
                depth = max(0, depth - 1)
            elif ch in TERMINALS and depth == 0:
                if i + 1 < len(line) and line[i + 1] in TERMINALS:
                    continue  # 連続する文末記号はまとめる
                flush(buf)
                buf = []
        flush(buf)
    return sentences
