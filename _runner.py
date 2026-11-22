"""テストランナー (harness.py から別プロセスで呼ばれる)

usage: python _runner.py <problem_dir> <solution.py> <public|all> <out.json> <timeout_sec>

problems/<id>/tests.py の規約:
  - def test_public_xxx(sol): ...   公開テスト (`harness.py test` で実行, 詳細表示)
  - def test_hidden_xxx(sol): ...   非公開テスト (提出時のみ実行)
  - sol は受験者の solution.py をモジュールとして import したもの
  - assert で検証する。assert のメッセージに期待値/実際値を書くと親切
"""
import contextlib
import importlib.util
import io
import json
import signal
import sys
import time
import traceback
from pathlib import Path


class TestTimeout(Exception):
    pass


def _alarm(signum, frame):
    raise TestTimeout("タイムアウト (計算量を見直してください)")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    problem_dir, sol_path, mode, out_path, timeout = sys.argv[1:6]
    problem_dir, sol_path, timeout = Path(problem_dir), Path(sol_path), int(timeout)
    sys.path.insert(0, str(sol_path.parent))
    out = {"error": None, "results": []}

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            sol = load_module("solution", sol_path)
    except Exception:
        out["error"] = traceback.format_exc(limit=-3)
        Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
        return

    tests = load_module("_tests", problem_dir / "tests.py")
    names = [n for n in vars(tests) if n.startswith("test_public_") or (mode == "all" and n.startswith("test_hidden_"))]
    signal.signal(signal.SIGALRM, _alarm)
    for name in names:
        fn = getattr(tests, name)
        kind = "public" if name.startswith("test_public_") else "hidden"
        t0 = time.perf_counter()
        passed, msg = True, ""
        try:
            signal.alarm(timeout)
            with contextlib.redirect_stdout(io.StringIO()):
                fn(sol)
        except AssertionError as e:
            passed = False
            tb = traceback.extract_tb(e.__traceback__)
            line = tb[-1].line if tb else ""
            msg = f"AssertionError: {e}" if str(e) else f"AssertionError at: {line}"
        except TestTimeout as e:
            passed, msg = False, str(e)
        except NotImplementedError:
            passed, msg = False, "NotImplementedError: 未実装です"
        except Exception:
            passed = False
            msg = traceback.format_exc(limit=-4)
        finally:
            signal.alarm(0)
        out["results"].append({
            "name": name.removeprefix("test_public_").removeprefix("test_hidden_"),
            "kind": kind, "passed": passed, "message": msg,
            "time_ms": (time.perf_counter() - t0) * 1000,
        })
    Path(out_path).write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
