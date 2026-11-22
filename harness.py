#!/usr/bin/env python3
"""NLP ライブコーディング対策ハーネス (自然言語処理)

使い方:
    python harness.py list                 # 問題一覧と成績
    python harness.py start [問題ID]        # 問題開始 (ID省略で次のおすすめ問題)。タイマー開始
    python harness.py start --random       # 未着手からランダム
    python harness.py start --category C   # カテゴリ指定
    python harness.py status               # 現在の問題と経過時間
    python harness.py test                 # 公開テストのみ実行 (タイマーは止まらない)
    python harness.py submit               # 提出 (タイマー停止 + 全テスト実行)
    python harness.py giveup               # ギブアップ (タイマー停止)
    python harness.py hint                 # ヒント使用回数を記録 (Claudeが使う)
    python harness.py review ID --quality N  # コード品質点を記録 (Claudeが使う)
    python harness.py stats                # 成績サマリ
    python harness.py selfcheck            # 全問題の参考解答でテストを検証 (開発用)

採点 (100点満点):
    テスト   60点 = 公開+非公開テストの通過率
    時間     20点 = 目標時間以内で満点, 目標の2倍で0点 (線形)
    コード品質 20点 = Claude のレビューによる (可読性/計算量/エッジケース/Pythonらしさ)
    ※ ヒント1回につき -3点 (最大 -15)
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROBLEMS = ROOT / "problems"
WORKSPACE = ROOT / "workspace"
HISTORY = ROOT / "history.json"
RUNNER = ROOT / "_runner.py"

CATEGORIES = {
    "A": "Python基礎・文字列処理",
    "B": "アルゴリズム (DP・探索)",
    "C": "古典NLP・情報検索・統計モデル",
    "D": "ライブラリ活用 (numpy/pandas/sklearn)",
    "E": "模擬本番 (60分総合問題)",
}

TEST_WEIGHT, TIME_WEIGHT, QUALITY_WEIGHT = 60, 20, 20
HINT_PENALTY, HINT_PENALTY_MAX = 3, 15


# ---------------------------------------------------------------- utilities
def now() -> datetime:
    return datetime.now().replace(microsecond=0)


def pad(s: str, width: int) -> str:
    """全角文字を幅2として左寄せ"""
    import unicodedata
    w = sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)
    return s + " " * max(0, width - w)


def fmt_sec(sec: float) -> str:
    sec = int(sec)
    return f"{sec // 60:02d}:{sec % 60:02d}"


def load_history() -> dict:
    if HISTORY.exists():
        return json.loads(HISTORY.read_text(encoding="utf-8"))
    return {"current": None, "attempts": []}


def save_history(h: dict) -> None:
    HISTORY.write_text(json.dumps(h, ensure_ascii=False, indent=2), encoding="utf-8")


def load_problems() -> list[dict]:
    probs = []
    for d in sorted(PROBLEMS.iterdir()):
        meta_path = d / "meta.json"
        if d.is_dir() and meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            meta["id"] = d.name
            meta["dir"] = d
            probs.append(meta)
    return probs


def find_problem(pid: str) -> dict:
    probs = load_problems()
    for p in probs:
        if p["id"] == pid or p["id"].split("_")[0] == pid.zfill(2):
            return p
    sys.exit(f"問題 '{pid}' が見つかりません。`python harness.py list` で確認してください。")


def best_by_problem(h: dict) -> dict[str, dict]:
    best: dict[str, dict] = {}
    for a in h["attempts"]:
        if a.get("status") not in ("submitted", "gaveup"):
            continue
        cur = best.get(a["problem"])
        if cur is None or (a.get("total") or 0) > (cur.get("total") or 0):
            best[a["problem"]] = a
    return best


def time_score(elapsed: float, target_min: float) -> float:
    target = target_min * 60
    if elapsed <= target:
        return float(TIME_WEIGHT)
    return round(max(0.0, TIME_WEIGHT * (1 - (elapsed - target) / target)), 1)


def recompute_total(a: dict) -> None:
    penalty = min(HINT_PENALTY * a.get("hints", 0), HINT_PENALTY_MAX)
    q = a.get("quality_score")
    a["hint_penalty"] = penalty
    if q is None:
        a["total"] = None
        a["provisional_total"] = round(a["test_score"] + a["time_score"] - penalty, 1)
    else:
        a["total"] = round(max(0.0, a["test_score"] + a["time_score"] + q - penalty), 1)
        a.pop("provisional_total", None)


def run_tests(problem: dict, solution: Path, mode: str) -> dict:
    """別プロセスでテスト実行。mode = 'public' | 'all'"""
    with tempfile.TemporaryDirectory() as td:
        out = Path(td) / "result.json"
        timeout = problem.get("timeout_sec", 10)
        cmd = [sys.executable, "-B", str(RUNNER), str(problem["dir"]), str(solution), mode, str(out), str(timeout)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout * 40 + 30,
                                  cwd=str(solution.parent))
        except subprocess.TimeoutExpired:
            return {"error": "テスト全体がタイムアウトしました", "results": []}
        if not out.exists():
            return {"error": (proc.stderr or proc.stdout)[-3000:], "results": []}
        return json.loads(out.read_text(encoding="utf-8"))


def print_results(res: dict, show_hidden_detail: bool = False) -> tuple[int, int, int, int]:
    if res.get("error"):
        print("\n[エラー] solution.py の読み込み/実行に失敗しました:\n")
        print(res["error"])
    pub_p = pub_t = hid_p = hid_t = 0
    for r in res["results"]:
        hidden = r["kind"] == "hidden"
        mark = "✅" if r["passed"] else "❌"
        label = "非公開" if hidden else "公開  "
        print(f"  {mark} [{label}] {r['name']}  ({r['time_ms']:.0f}ms)")
        if not r["passed"]:
            msg = r["message"]
            if hidden and not show_hidden_detail:
                msg = msg.splitlines()[-1][:200] if msg else ""
            for line in msg.splitlines()[-15:]:
                print(f"        {line}")
        if hidden:
            hid_t += 1
            hid_p += r["passed"]
        else:
            pub_t += 1
            pub_p += r["passed"]
    return pub_p, pub_t, hid_p, hid_t


# ---------------------------------------------------------------- commands
def cmd_list(args) -> None:
    h = load_history()
    best = best_by_problem(h)
    tried = {a["problem"] for a in h["attempts"]}
    cur_cat = None
    for p in load_problems():
        if p["category"] != cur_cat:
            cur_cat = p["category"]
            print(f"\n== [{cur_cat}] {CATEGORIES.get(cur_cat, '')} ==")
        b = best.get(p["id"])
        if b:
            score = b["total"] if b["total"] is not None else f"{b['provisional_total']}(暫定)"
            st = f"best {score}点 / {fmt_sec(b['elapsed_sec'])}"
        elif p["id"] in tried:
            st = "挑戦中"
        else:
            st = "未着手"
        print(f"  {p['id']:<24} {'★' * p['difficulty']:<5} 目標{p['target_minutes']:>3}分  {pad(p['title'], 46)} {st}")
    print()


def cmd_start(args) -> None:
    h = load_history()
    if h["current"] and not args.force:
        c = h["current"]
        el = (now() - datetime.fromisoformat(c["started_at"])).total_seconds()
        sys.exit(f"現在 {c['problem']} に挑戦中です (経過 {fmt_sec(el)})。\n"
                 f"`python harness.py submit` か `giveup` で終了してください (強制開始は --force)。")
    if h["current"]:  # --force: 挑戦中のものは放棄扱い
        for x in h["attempts"]:
            if x["id"] == h["current"]["attempt_id"]:
                x["status"] = "abandoned"
        h["current"] = None
    probs = load_problems()
    if args.problem:
        prob = find_problem(args.problem)
    else:
        tried = {a["problem"] for a in h["attempts"] if a.get("status") in ("submitted", "gaveup")}
        cands = [p for p in probs if p["id"] not in tried]
        if args.category:
            cands = [p for p in cands if p["category"] == args.category.upper()]
        if not cands:
            sys.exit("未着手の問題がありません。問題IDを指定して再挑戦してください。")
        prob = random.choice(cands) if args.random else cands[0]

    attempt_id = len(h["attempts"]) + 1
    wdir = WORKSPACE / f"{attempt_id:03d}_{prob['id']}"
    wdir.mkdir(parents=True, exist_ok=False)
    shutil.copy(prob["dir"] / "template.py", wdir / "solution.py")
    shutil.copy(prob["dir"] / "problem.md", wdir / "problem.md")
    for extra in prob.get("data_files", []):
        shutil.copy(prob["dir"] / extra, wdir / extra)

    started = now()
    h["current"] = {"attempt_id": attempt_id, "problem": prob["id"], "started_at": started.isoformat(),
                    "workdir": str(wdir.relative_to(ROOT)), "hints": 0}
    h["attempts"].append({"id": attempt_id, "problem": prob["id"], "category": prob["category"],
                          "status": "in_progress", "started_at": started.isoformat(),
                          "target_minutes": prob["target_minutes"], "workdir": str(wdir.relative_to(ROOT))})
    save_history(h)

    print("=" * 72)
    print((prob["dir"] / "problem.md").read_text(encoding="utf-8"))
    print("=" * 72)
    print(f"⏱  タイマー開始: {started:%H:%M:%S}   目標時間: {prob['target_minutes']}分")
    print(f"📝 解答ファイル: {wdir.relative_to(ROOT)}/solution.py")
    print("   公開テスト: python harness.py test   /   提出: python harness.py submit")


def _current_or_exit(h: dict) -> dict:
    if not h["current"]:
        sys.exit("挑戦中の問題はありません。`python harness.py start` で開始してください。")
    return h["current"]


def cmd_status(args) -> None:
    h = load_history()
    c = _current_or_exit(h)
    prob = find_problem(c["problem"])
    el = (now() - datetime.fromisoformat(c["started_at"])).total_seconds()
    tgt = prob["target_minutes"] * 60
    bar_len = 30
    filled = min(bar_len, int(bar_len * el / tgt))
    over = "  ⚠ 目標超過!" if el > tgt else ""
    print(f"問題: {c['problem']}  ({prob['title']})")
    print(f"経過: {fmt_sec(el)} / 目標 {fmt_sec(tgt)}  [{'#' * filled}{'.' * (bar_len - filled)}]{over}")
    print(f"ヒント使用: {c.get('hints', 0)}回   解答: {c['workdir']}/solution.py")


def cmd_test(args) -> None:
    h = load_history()
    c = _current_or_exit(h)
    prob = find_problem(c["problem"])
    sol = ROOT / c["workdir"] / "solution.py"
    el = (now() - datetime.fromisoformat(c["started_at"])).total_seconds()
    print(f"[公開テスト] {c['problem']}  経過 {fmt_sec(el)} / 目標 {prob['target_minutes']}分\n")
    res = run_tests(prob, sol, "public")
    p, t, _, _ = print_results(res, show_hidden_detail=True)
    print(f"\n公開テスト: {p}/{t} 通過  (提出時は非公開テストも実行されます)")


def _finish(status: str) -> None:
    h = load_history()
    c = _current_or_exit(h)
    prob = find_problem(c["problem"])
    sol = ROOT / c["workdir"] / "solution.py"
    end = now()
    elapsed = (end - datetime.fromisoformat(c["started_at"])).total_seconds()

    print(f"[提出] {c['problem']}  所要時間 {fmt_sec(elapsed)} / 目標 {prob['target_minutes']}分\n")
    res = run_tests(prob, sol, "all")
    pub_p, pub_t, hid_p, hid_t = print_results(res)
    total_t = pub_t + hid_t
    ratio = (pub_p + hid_p) / total_t if total_t else 0.0

    a = next(x for x in h["attempts"] if x["id"] == c["attempt_id"])
    a.update({
        "status": status,
        "submitted_at": end.isoformat(),
        "elapsed_sec": int(elapsed),
        "public": [pub_p, pub_t],
        "hidden": [hid_p, hid_t],
        "test_score": round(TEST_WEIGHT * ratio, 1),
        "time_score": time_score(elapsed, prob["target_minutes"]) if status == "submitted" else 0.0,
        "hints": c.get("hints", 0),
        "quality_score": None,
        "error": res.get("error"),
    })
    recompute_total(a)
    h["current"] = None
    save_history(h)

    print("\n" + "-" * 60)
    print(f"テスト:   {a['test_score']:>5} / {TEST_WEIGHT}  (公開 {pub_p}/{pub_t}, 非公開 {hid_p}/{hid_t})")
    print(f"時間:     {a['time_score']:>5} / {TIME_WEIGHT}  ({fmt_sec(elapsed)} / 目標 {prob['target_minutes']}分)")
    if a["hint_penalty"]:
        print(f"ヒント:   -{a['hint_penalty']}")
    print(f"品質:     (Claudeのレビュー待ち) / {QUALITY_WEIGHT}")
    print(f"暫定合計: {a['provisional_total']} / {TEST_WEIGHT + TIME_WEIGHT}")
    print("-" * 60)
    print(f"attempt #{a['id']} を記録しました。Claude に「採点して」と伝えるとレビューします。")


def cmd_submit(args) -> None:
    _finish("submitted")


def cmd_giveup(args) -> None:
    _finish("gaveup")


def cmd_hint(args) -> None:
    h = load_history()
    c = _current_or_exit(h)
    c["hints"] = c.get("hints", 0) + 1
    save_history(h)
    print(f"ヒント使用回数: {c['hints']} (1回につき -{HINT_PENALTY}点, 最大 -{HINT_PENALTY_MAX})")


def cmd_review(args) -> None:
    h = load_history()
    a = next((x for x in h["attempts"] if x["id"] == args.attempt_id), None)
    if a is None or a["status"] == "in_progress":
        sys.exit("その attempt は存在しないか、まだ提出されていません。")
    a["quality_score"] = max(0.0, min(float(QUALITY_WEIGHT), args.quality))
    review_file = ROOT / a["workdir"] / "review.md"
    if review_file.exists():
        a["review_file"] = str(review_file.relative_to(ROOT))
    recompute_total(a)
    save_history(h)
    print(f"attempt #{a['id']} ({a['problem']}): 品質 {a['quality_score']}/{QUALITY_WEIGHT} → 合計 {a['total']}/100")


def cmd_stats(args) -> None:
    h = load_history()
    done = [a for a in h["attempts"] if a["status"] in ("submitted", "gaveup")]
    if not done:
        print("まだ提出がありません。")
        return
    print(f"{'#':>3} {'問題':<28} {'状態':<6} {'時間':>6} {'目標':>4} {'テスト':>6} {'時間点':>5} {'品質':>5} {'合計':>6}")
    for a in done:
        tot = a["total"] if a["total"] is not None else f"{a['provisional_total']}*"
        q = a["quality_score"] if a["quality_score"] is not None else "-"
        st = "提出" if a["status"] == "submitted" else "断念"
        print(f"{a['id']:>3} {a['problem']:<28} {st:<6} {fmt_sec(a['elapsed_sec']):>6} {a['target_minutes']:>4} "
              f"{a['test_score']:>6} {a['time_score']:>5} {q:>5} {tot:>6}")
    print("  (* = 品質レビュー前の暫定値)\n")
    print("カテゴリ別:")
    for cat, name in CATEGORIES.items():
        xs = [a for a in done if a["category"] == cat]
        if not xs:
            continue
        avg_test = sum(a["test_score"] for a in xs) / len(xs) / TEST_WEIGHT * 100
        ratio = sum(a["elapsed_sec"] / (a["target_minutes"] * 60) for a in xs) / len(xs)
        print(f"  [{cat}] {name:<28} {len(xs)}回  テスト通過率 {avg_test:5.1f}%  目標時間比 {ratio:4.2f}x")
    total_time = sum(a["elapsed_sec"] for a in done)
    print(f"\n総練習時間: {fmt_sec(total_time)} ({total_time / 3600:.1f}時間)  "
          f"解いた問題: {len({a['problem'] for a in done})}/{len(load_problems())}")


def cmd_selfcheck(args) -> None:
    ok = True
    for p in load_problems():
        if args.problem and p["id"] != find_problem(args.problem)["id"]:
            continue
        ref = p["dir"] / "reference.py"
        res = run_tests(p, ref, "all")
        n = len(res["results"])
        passed = sum(r["passed"] for r in res["results"])
        pub = sum(r["kind"] == "public" for r in res["results"])
        good = not res.get("error") and passed == n and n > 0
        # テンプレートが全テストを通過してしまわないか
        tres = run_tests(p, p["dir"] / "template.py", "all")
        tpassed = sum(r["passed"] for r in tres["results"])
        print(f"{'OK ' if good else 'NG '} {p['id']:<28} reference {passed}/{n} (公開{pub})  template {tpassed}/{n}")
        if not good:
            ok = False
            print_results(res, show_hidden_detail=True)
    sys.exit(0 if ok else 1)


def main() -> None:
    ap = argparse.ArgumentParser(description="NLP ライブコーディング対策ハーネス")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(func=cmd_list)
    s = sub.add_parser("start")
    s.add_argument("problem", nargs="?")
    s.add_argument("--random", action="store_true")
    s.add_argument("--category")
    s.add_argument("--force", action="store_true")
    s.set_defaults(func=cmd_start)
    sub.add_parser("status").set_defaults(func=cmd_status)
    sub.add_parser("test").set_defaults(func=cmd_test)
    sub.add_parser("submit").set_defaults(func=cmd_submit)
    sub.add_parser("giveup").set_defaults(func=cmd_giveup)
    sub.add_parser("hint").set_defaults(func=cmd_hint)
    r = sub.add_parser("review")
    r.add_argument("attempt_id", type=int)
    r.add_argument("--quality", type=float, required=True)
    r.set_defaults(func=cmd_review)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sc = sub.add_parser("selfcheck")
    sc.add_argument("problem", nargs="?")
    sc.set_defaults(func=cmd_selfcheck)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
