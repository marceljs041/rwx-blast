"""Build time: run every test of the pinned commit once, in a fork of the warm process, and record what it
touches. Everything written here describes the pinned commit only: lines executed, classes instantiated, files
opened, CPU used, and DuckDB answers to the fixture queries."""
import ast
import importlib
import importlib.util
import json
import os
import pickle
import subprocess
import sys
import time
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from br import common as C
from br import keytrack
from br import stmts

ROOT_S = C.ROOT + "/"
# Test files whose tests are run against recorded DuckDB answers.
MEMO_FILES = ("tests/test_executor.py", "tests/test_optimizer.py")
mon = sys.monitoring
TID = 3
TID2 = 4


_start_lines = {}


def start_line(code, off):
    """The line a call of `code` starts on. The tracer reports a line only when execution moves to a different
    one, so a function whose body sits on its first line (a one-line lambda or def) never reports any line."""
    line = _start_lines.get(code)
    if line is None:
        line = code.co_firstlineno
        for a, b, ln in code.co_lines():
            if a <= off < b and ln is not None:
                line = ln
                break
        _start_lines[code] = line
    return line


def start_line_trace(sink, funcs=None, skip=None):
    """Collects (file, line) of sqlglot code as it runs: every line reported by the tracer, and the first
    line of every code object that is entered. `sink` is a one-item list holding the set to add to."""
    def on_line(code, line):
        fn = code.co_filename
        if fn.startswith(ROOT_S) and not (skip and skip(code)):
            sink[0].add((fn, line))
            if funcs is not None and code.co_flags & 1:
                funcs.add((fn[len(ROOT_S):], code.co_firstlineno, code.co_name))
        return mon.DISABLE

    def on_enter(code, off):
        fn = code.co_filename
        if fn.startswith(ROOT_S) and not (skip and skip(code)):
            sink[0].add((fn, start_line(code, off)))
            if funcs is not None and code.co_flags & 1:
                funcs.add((fn[len(ROOT_S):], code.co_firstlineno, code.co_name))
        return mon.DISABLE

    mon.use_tool_id(TID, "br")
    mon.register_callback(TID, mon.events.LINE, on_line)
    mon.set_events(TID, mon.events.LINE)
    mon.use_tool_id(TID2, "br-enter")
    mon.register_callback(TID2, mon.events.PY_START, on_enter)
    mon.set_events(TID2, mon.events.PY_START)


def stop_trace():
    mon.set_events(TID, 0)
    mon.register_callback(TID, mon.events.LINE, None)
    mon.free_tool_id(TID)
    mon.set_events(TID2, 0)
    mon.register_callback(TID2, mon.events.PY_START, None)
    mon.free_tool_id(TID2)


def init_codes():
    """Code objects of every __init__ / __new__ defined on a sqlglot class."""
    out = set()
    seen = set()

    def visit(c):
        if c in seen:
            return
        seen.add(c)
        for k, v in list(vars(c).items()):
            if k in ("__init__", "__new__"):
                f = getattr(v, "__func__", v)
                code = getattr(f, "__code__", None)
                if code is not None:
                    out.add(code)
            elif isinstance(v, type):
                visit(v)

    for name, m in list(sys.modules.items()):
        if m is not None and (name == "sqlglot" or name.startswith("sqlglot.")):
            for v in list(vars(m).values()):
                if isinstance(v, type) and (v.__module__ or "").startswith("sqlglot"):
                    visit(v)
    return out


def traced(fn, inits, per_call=False):
    """Runs fn() while recording lines, instantiated classes and opened paths. With per_call, lines are also
    recorded separately for every pool call the test makes."""
    lines = set()
    types_seen = set()
    opened = set()
    call_lines = {}   # (file, line) -> bitmask of calls
    call_lines2 = {}  # the same, for the later stage of staged calls only
    call_lines3 = {}  # the same, for the later stage outside the executor's code generator calls
    gen_rec = {}      # k -> the code generator calls made by the later stage of call k
    check_lines = set()  # lines the run-time replay of generator calls runs besides the calls themselves
    state = {"sink": lines, "cur": None, "types": None, "t0": 0.0, "gk": None, "gdepth": 0}
    call_types = {}   # k -> types instantiated inside that call
    setup_types = set()
    rest_types = set()   # instantiated outside every unit: such objects can be used by any statement unit
    unit_lines = set()   # lines run inside any unit; `lines` itself only collects what runs outside units
    stmt_lines = {}   # (file, line) -> bitmask of statement units
    stmt_types = {}
    stmt_cpu = {}

    def on_start(code, off):
        loc = sys._getframe(1).f_locals
        s = loc.get("self")
        if s is not None:
            t = type(s)
        else:
            t = loc.get("cls")
            if not isinstance(t, type):
                return
        types_seen.add(t)
        if state["types"] is not None:
            state["types"].add(t)
        else:
            rest_types.add(t)

    def hook(event, args):
        if event == "open" or event == "os.listdir" or event == "os.scandir":
            p = args[0]
            if isinstance(p, bytes):
                p = os.fsdecode(p)
            if isinstance(p, str):
                p = os.path.abspath(p)
                if p.startswith(ROOT_S):
                    opened.add(p[len(ROOT_S):] + ("/" if event != "open" else ""))

    def on_call(phase, k):
        if phase == 2:                       # pool initializer: state every later call can see
            state["types"] = setup_types
            return
        if phase == 3:
            state["types"] = None
            return
        if phase == 4:                       # first stage of a staged call is over
            state["cur2"] = set()
            state["gin"] = set()
            state["gk"] = k
            gen_rec[k] = dict(ok=True, cfgs=[], blobs=[], ids={}, gens={}, seq=[])
            trace_into(state["cur2"])
            mon.restart_events()
            return
        if phase == 0:
            state["cur"] = set()
            state["cur2"] = None
            state["gk"] = None
            state["types"] = call_types.setdefault(k, set())
            trace_into(state["cur"])
        else:
            state["types"] = None
            state["gk"] = None
            if state["cur2"] is not None:
                bit2 = 1 << k
                for key in state["cur2"]:
                    call_lines3[key] = call_lines3.get(key, 0) | bit2
                state["cur2"] |= state["gin"]
                for key in state["cur2"]:
                    call_lines2[key] = call_lines2.get(key, 0) | bit2
                state["cur"] |= state["cur2"]
                state["cur2"] = None
            bit = 1 << k
            for key in state["cur"]:
                call_lines[key] = call_lines.get(key, 0) | bit
            unit_lines.update(state["cur"])
            trace_into(lines)
        mon.restart_events()

    def on_unit(phase, k):
        if phase == 0:
            state["cur"] = set()
            state["types"] = stmt_types.setdefault(k, set())
            trace_into(state["cur"])
            mon.restart_events()
            state["t0"] = time.process_time()
        else:
            stmt_cpu[k] = stmt_cpu.get(k, 0.0) + time.process_time() - state["t0"]
            state["types"] = None
            bit = 1 << k
            for key in state["cur"]:
                stmt_lines[key] = stmt_lines.get(key, 0) | bit
            unit_lines.update(state["cur"])
            trace_into(lines)
            mon.restart_events()

    sink = [lines]

    def trace_into(target):
        sink[0] = target

    # The later stage of a staged pool call turns expressions into Python code with the executor's generator.
    # Each such call is recorded (input, output, which generator object), and the lines it runs are kept apart
    # from the rest of the later stage: when a change is seen only inside these calls, replaying them at run
    # time decides whether the later stage can behave differently.
    G = sys.modules.get("sqlglot.generator") if per_call else None
    real_gen = vars(G.Generator).get("generate") if G is not None else None

    def generate(self, expression, copy=True):
        k = state["gk"]
        fut = C.FC.cur
        if k is None or state["gdepth"] or fut is None or fut._k != k or stmts.SU.active:
            return real_gen(self, expression, copy)
        rec = gen_rec[k]
        prev = sink[0]
        state["gdepth"] += 1
        oi = inst = None
        try:
            if rec["ok"]:
                trace_into(check_lines)
                mon.restart_events()
                try:
                    if copy is not True:
                        rec["ok"] = False
                    blob = pickle.dumps(expression, protocol=5)
                    oi = rec["ids"].get(blob)
                    if oi is None:
                        oi = rec["ids"][blob] = len(rec["blobs"])
                        rec["blobs"].append(blob)
                    if id(self) not in rec["gens"]:
                        rec["gens"][id(self)] = (len(rec["gens"]), self)
                        rec["cfgs"].append(C.gen_cfg(self))
                    inst = rec["gens"][id(self)][0]
                except Exception:
                    rec["ok"] = False
            trace_into(state["gin"])
            mon.restart_events()
            try:
                out = real_gen(self, expression, copy)
            except BaseException:
                rec["ok"] = False
                raise
            if rec["ok"]:
                rec["seq"].append((inst, oi, out))
            return out
        finally:
            state["gdepth"] -= 1
            trace_into(prev)
            mon.restart_events()

    # Modules asked for by name, through import statements, __import__ or importlib.import_module, recorded as
    # "import:<module>" among the opened paths: a test that asks for a module needs it to import.
    import builtins
    import importlib as _il
    real_import = builtins.__import__
    real_import_module = _il.import_module

    def note(name):
        if name.startswith("sqlglot"):
            opened.add("import:" + name)

    def imp(name, globals=None, locals=None, fromlist=(), level=0):
        try:
            full = name
            if level:
                full = _il._bootstrap._resolve_name(name, (globals or {}).get("__package__") or "", level)
            note(full)
            for f in fromlist or ():
                if isinstance(f, str) and f != "*":
                    note(f"{full}.{f}")
        except Exception:
            pass
        return real_import(name, globals, locals, fromlist, level)

    def import_module(name, package=None):
        try:
            note(_il.util.resolve_name(name, package) if name.startswith(".") else name)
        except Exception:
            pass
        return real_import_module(name, package)

    sys.addaudithook(hook)
    start_line_trace(sink)
    mon.restart_events()   # lines switched off while tracing the imports must report again
    if per_call:
        C.FC.hook = on_call
        C.FC.record = {}
        stmts.SU.hook = on_unit
        if real_gen is not None:
            G.Generator.generate = generate
    mon.register_callback(TID, mon.events.PY_START, on_start)
    for code in inits:
        mon.set_local_events(TID, code, mon.events.PY_START)
    t0 = time.process_time()
    builtins.__import__ = imp
    _il.import_module = import_module
    try:
        result = fn()
    finally:
        builtins.__import__ = real_import
        _il.import_module = real_import_module
        if real_gen is not None:
            G.Generator.generate = real_gen
    cpu = time.process_time() - t0
    # Replays every recorded set of generator calls on the pinned code, as the run time would: only a set that
    # reproduces itself is kept. The replay's own lines (unpickling, making generators) are recorded: a change
    # to any of them rules the replay out.
    gens_out = {}
    scratch = set()

    def on_part(part):
        trace_into(check_lines if part == "check" else scratch)
        mon.restart_events()

    for k, rec in gen_rec.items():
        if rec["ok"] and rec["seq"] and k in (C.FC.record or {}):
            g = (rec["cfgs"], rec["blobs"], rec["seq"])
            if C.gens_match(g, on_part):
                gens_out[k] = g
        if os.environ.get("BR_GEN_DEBUG"):  # build diagnostics
            sys.stderr.write(f"GEN k={k} ok={rec['ok']} seq={len(rec['seq'])} gens={len(rec['cfgs'])} "
                             f"rec={k in (C.FC.record or {})} kept={k in gens_out}\n")
    trace_into(lines)
    mon.set_events(TID, 0)
    mon.set_events(TID2, 0)
    def keys_of(ts):
        out = set()
        for t in ts:
            for k in t.__mro__:
                if (k.__module__ or "").startswith("sqlglot"):
                    out.add(f"{k.__module__}:{k.__qualname__}")
        return out

    classes = keys_of(types_seen)
    calls = None
    if per_call and (C.FC.record or stmt_cpu):
        def class_masks(by_unit):
            out = {}
            for k, ts in by_unit.items():
                for key in keys_of(ts):
                    out[key] = out.get(key, 0) | (1 << k)
            return out
        calls = dict(record=C.FC.record, lines=call_lines, lines2=call_lines2, lines3=call_lines3, gens=gens_out,
                     gen_check=check_lines, classes=class_masks(call_types),
                     rest_lines=set(lines),
                     setup_classes=keys_of(setup_types), rest_classes=keys_of(rest_types),
                     stmt_lines=stmt_lines, stmt_classes=class_masks(stmt_types), stmt_cpu=stmt_cpu)
    return dict(result=result, cpu=cpu, lines=lines | unit_lines, classes=classes, opened=opened, calls=calls)


def run_pool(jobs, workers, tmpdir):
    """jobs: list of (key, callable). Each runs in a fork; returns {key: value or None}."""
    out = {}
    live = {}
    queue = list(jobs)
    n = 0
    while queue or live:
        while queue and len(live) < workers:
            key, fn = queue.pop()
            n += 1
            path = os.path.join(tmpdir, f"job{n}.pkl")
            pid = os.fork()
            if pid == 0:
                code = 0
                try:
                    with open(path, "wb") as f:
                        pickle.dump(fn(), f)
                except BaseException as e:
                    C.log("job failed", key, repr(e)[:300])
                    code = 1
                os._exit(code)
            live[pid] = (key, path)
        pid, _ = os.waitpid(-1, 0)
        if pid in live:
            key, path = live.pop(pid)
            try:
                with open(path, "rb") as f:
                    out[key] = pickle.load(f)
                os.unlink(path)
            except Exception:
                out[key] = None
    return out


def key_pass(ids, seconds, workers, tmpdir, go_fd, out_path):
    """Runs in its own process, forked before any test module is imported: replaces dict tables by tracking
    copies and runs every test once more, noting which entries it reads. One job per test file imports the
    file and sets its classes up, then forks once per test; reads made by the import and the set-ups count for
    every test forked after them. Files whose tests need the recorded DuckDB answers wait for `go_fd`; the
    others run while the parent process records those answers on one core."""
    stop_trace()
    t0 = time.time()
    os.makedirs(tmpdir, exist_ok=True)
    spare = max(1, workers - 1)

    # Code that runs while a test file is imported (a dialect defined in the file, a constant computed by a
    # helper) runs for every test of that file, but inside no test. Module and class bodies of the test files
    # themselves are left out: every test of a file "runs" those.
    tests_dir = ROOT_S + "tests/"

    def import_job(mod):
        def run():
            lines = set()
            start_line_trace([lines], skip=lambda c: c.co_filename.startswith(tests_dir) and not c.co_flags & 1)
            mon.restart_events()
            try:
                importlib.import_module(mod)
            except BaseException:
                pass
            mon.set_events(TID, 0)
            mon.set_events(TID2, 0)
            return sorted(lines)
        return run

    by_file = {}
    for i, tid in enumerate(ids):
        by_file.setdefault(tid.split("::")[0], []).append((i, tid))
    test_files = sorted(by_file)
    modlines = run_pool([(f, import_job(f[:-3].replace("/", "."))) for f in test_files], spare, tmpdir)
    for name in ("numpy", "pandas", "duckdb"):      # not sqlglot code: no need to import them once per file
        try:
            __import__(name)
        except Exception:
            pass
    table = keytrack.install()
    R = C.Runner()
    R.instrument = False

    def file_job(members):
        def run():
            keytrack.reset()
            out = {}
            for i, tid in members:
                try:
                    c, _ = R.get_class(tid)
                    R.ensure_setup(c)
                except BaseException:
                    pass
                path = os.path.join(tmpdir, f"t{i}.pkl")
                pid = os.fork()
                if pid == 0:
                    code = 0
                    try:
                        ok = bool(R.run(tid)[0])
                        with open(path, "wb") as f:
                            pickle.dump((ok, keytrack.dump()), f, protocol=5)
                    except BaseException:
                        code = 1
                    os._exit(code)
                os.waitpid(pid, 0)
                try:
                    with open(path, "rb") as f:
                        out[i] = pickle.load(f)
                    os.unlink(path)
                except Exception:
                    out[i] = None
            return out
        return run

    def chunks(files):
        """Jobs of about equal test time; a long file is split, each part importing the file itself."""
        out = []
        for f in files:
            part, t = [], 0.0
            for i, tid in by_file[f]:
                part.append((i, tid))
                t += seconds[i]
                if t >= 6.0:
                    out.append((t, part))
                    part, t = [], 0.0
            if part:
                out.append((t, part))
        out.sort(key=lambda x: x[0])               # run_pool starts from the end: longest first
        return [((p[0][0], len(p)), file_job(p)) for _, p in out]

    late = [f for f in test_files if f in MEMO_FILES]
    early = [f for f in test_files if f not in MEMO_FILES]
    res = {}
    if table:
        for r in run_pool(chunks(early), spare, tmpdir).values():
            res.update(r or {})
    os.read(go_fd, 1)                       # the DuckDB answers are on disk now
    try:
        with open(os.path.join(C.MAPS, "duck.pkl"), "rb") as f:
            memo = pickle.load(f)
    except Exception:
        memo = {}
    R.after_setup = lambda c: C.wrap_executor(c, memo)
    if table:
        for r in run_pool(chunks(late), workers, tmpdir).values():
            res.update(r or {})
    with open(out_path, "wb") as f:
        pickle.dump((table, res, modlines), f, protocol=5)
    C.log(f"import of test files: {sum(1 for v in modlines.values() if v)} of {len(test_files)} run sqlglot code, "
          f"{sum(len(v) for v in modlines.values() if v)} lines")
    C.log(f"key pass: {len(table)} class attributes and globals hold {len(set(table.values()))} tracked tables, "
          f"{sum(1 for r in res.values() if r and r[0])} of {len(ids)} tests passed with tracking, "
          f"{time.time() - t0:.1f}s")


def test_import_graph():
    """tests module name -> test files that import it, directly or through other test modules."""
    deps = {}
    base = os.path.join(C.ROOT, "tests")
    for d, _, fs in os.walk(base):
        for f in fs:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(d, f), C.ROOT)
            mod = rel[:-3].replace("/", ".")
            if mod.endswith(".__init__"):
                mod = mod[:-9]
            used = set()
            try:
                tree = ast.parse(open(os.path.join(d, f), encoding="utf-8").read())
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == "tests":
                    used.add(node.module)
                    for a in node.names:
                        used.add(node.module + "." + a.name)
                elif isinstance(node, ast.Import):
                    for a in node.names:
                        if a.name.split(".")[0] == "tests":
                            used.add(a.name)
            deps[mod] = (rel, used)
    importers = {m: {deps[m][0]} for m in deps}
    changed = True
    while changed:
        changed = False
        for m, (rel, used) in deps.items():
            for u in used:
                if u in importers:
                    add = importers[m] - importers[u]
                    if add:
                        importers[u] |= add
                        changed = True
    return {m: sorted(v) for m, v in importers.items()}


def main():
    out_dir = C.MAPS
    os.makedirs(out_dir, exist_ok=True)
    tmpdir = os.path.join(C.TMP, "br_build")
    os.makedirs(tmpdir, exist_ok=True)
    tests = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "tests_baseline.json")))["tests"]
    ids = [t["id"] for t in tests]
    # The build has 4 cores and 8 GB; os.cpu_count() reports the host's cores, so it is not used.
    workers = int(os.environ.get("BR_JOBS", 4))
    t_start = time.time()

    core = subprocess.run(
        [sys.executable, "-c",
         "import sys, sqlglot; print('\\n'.join(m for m in sys.modules if m.split('.')[0] == 'sqlglot'))"],
        capture_output=True, text=True, cwd=C.ROOT, env=dict(os.environ, PYTHONPATH=C.ROOT),
    ).stdout.split()

    import_lines = set()
    import_funcs = set()
    start_line_trace([import_lines], import_funcs)
    C.warm()
    # The pass that records table reads starts from here, before any test module is imported, and waits.
    key_path = os.path.join(tmpdir, "keys.pkl")
    go_r, go_w = os.pipe()
    key_pid = os.fork() if os.environ.get("BR_KEYS", "1") != "0" else -1
    if key_pid == 0:
        code = 0
        try:
            os.close(go_w)
            key_pass(ids, [t["seconds"] for t in tests], workers, os.path.join(tmpdir, "keys"), go_r, key_path)
        except BaseException as e:
            C.log("key pass failed", repr(e)[:300])
            code = 1
        os._exit(code)
    os.close(go_r)
    R = C.Runner()
    for tid in ids:
        try:
            R.get_class(tid)
        except Exception as e:
            C.log("cannot import", tid, repr(e)[:200])
    stop_trace()
    inits = init_codes()
    C.log(f"warm: {len(import_lines)} import lines, {len(inits)} constructors, {time.time() - t_start:.1f}s")

    # DuckDB answers for the fixture queries of the executor tests.
    memo_tests = [i for i in ids if (i.startswith("tests/test_executor.py::TestExecutor::")
                                     and i.rsplit("::", 1)[1] in ("test_execute_tpcds", "test_execute_tpch",
                                                                  "test_optimized_tpch"))
                  or i.startswith("tests/test_optimizer.py::TestOptimizer::")]

    def record_memo(part):
        def run():
            memo = {}
            R.after_setup = lambda c: C.wrap_executor(c, memo, record=True)
            res = [R.run(i) for i in part]
            return memo, res
        return run

    # Answers are keyed by test class and query text, so parts recorded in separate processes simply merge.
    sec_of = {t["id"]: t["seconds"] for t in tests}
    parts = [[0.0, []] for _ in range(max(1, workers))]
    for tid in sorted(memo_tests, key=lambda x: -sec_of.get(x, 0.0)):
        slot = min(parts, key=lambda p: p[0])
        slot[0] += sec_of.get(tid, 0.0)
        slot[1].append(tid)
    memo, memo_res = {}, []
    for r in run_pool([(k, record_memo(p[1])) for k, p in enumerate(parts) if p[1]], workers, tmpdir).values():
        if r is None:
            C.log("a DuckDB recording job failed")
            continue
        for sec, answers in r[0].items():
            mine = memo.setdefault(sec, {})
            for sql, slot in answers.items():
                mine.setdefault(sql, {}).update(slot)      # a slot holds the answer as rows, as a frame or both
        memo_res.extend(r[1])
    C.log(f"duck memo: {sum(len(m) for m in memo.values())} queries, "
          f"{sum(not r[0] for r in memo_res)} of {len(memo_res)} tests failed while recording, "
          f"{time.time() - t_start:.1f}s")
    memo = {sec: {sql: slot for sql, slot in m.items() if slot} for sec, m in memo.items()}
    with open(os.path.join(out_dir, "duck.pkl"), "wb") as f:
        pickle.dump(memo, f, protocol=5)
    R.after_setup = lambda c: C.wrap_executor(c, memo)

    key_table, key_res, modlines = {}, {}, {}
    if key_pid > 0:
        os.write(go_w, b"g")
        os.close(go_w)
        try:
            os.waitpid(key_pid, 0)
            with open(key_path, "rb") as f:
                key_table, key_res, modlines = pickle.load(f)
            os.unlink(key_path)
        except Exception as e:
            C.log("no key maps", repr(e)[:200])
            key_table, key_res, modlines = {}, {}, {}

    # Import state that differs between two imports of the unchanged tree (order of set iteration).
    def unstable_job():
        from br import statediff
        before = statediff.snapshot()
        ov = os.path.join(tmpdir, f"ov_{os.getpid()}")
        C.build_overlay(C.ROOT, ov, {})
        C.reimport(ov)
        after = statediff.snapshot()
        keys = statediff.diff(before, after)
        return [k for k in keys if k in before and k in after], [k for k in keys if not (k in before and k in after)]

    unstable = set()
    lazy = set()
    for r in run_pool([(i, unstable_job) for i in range(4)], workers, tmpdir).values():
        if r:
            unstable.update(r[0])
            lazy.update(r[1])
    C.log(f"state keys with unstable order: {len(unstable)}, lazily bound: {len(lazy)}")

    # Per-test traces.
    def trace_job(tid):
        def run():
            # Objects made by the class set-up live on while the test's pool calls run.
            C.FC.hook and C.FC.hook(2, -1)
            try:
                c, _ = R.get_class(tid)
                R.ensure_setup(c)
            except BaseException:
                pass
            C.FC.hook and C.FC.hook(3, -1)
            return R.run(tid)
        return lambda: traced(run, inits, per_call=True)

    traces = run_pool([(i, trace_job(tid)) for i, tid in enumerate(ids)], workers, tmpdir)
    C.log(f"traced {len(traces)} tests, {time.time() - t_start:.1f}s")

    # Class set-up traces.
    classes = {}
    for tid in ids:
        try:
            c, _ = R.get_class(tid)
            classes[tid.rsplit("::", 1)[0]] = c
        except Exception:
            pass

    def setup_job(c):
        def run():
            R.ensure_setup(c)
            return R.setup.get(c)
        return lambda: traced(run, inits)

    setups = run_pool([(k, setup_job(c)) for k, c in classes.items()], workers, tmpdir)

    # CPU of the longer tests without tracing.
    def cpu_job(tid):
        def run():
            c, _ = R.get_class(tid)
            R.ensure_setup(c)                 # class set-up is paid once per class, not per test
            t0 = time.process_time()
            ok = R.run(tid)[0]
            return ok, time.process_time() - t0
        return run

    # A test whose statements depend on each other fails when half of them are left out: no skipping for it.
    def half_job(tid, parity, n):
        def run():
            stmts.SU.affected = {k for k in range(n) if k % 2 == parity}
            return R.run(tid)[0]
        return run

    half_jobs = []
    for i, tid in enumerate(ids):
        tr = traces.get(i)
        scpu = tr and tr.get("calls") and tr["calls"]["stmt_cpu"]
        if scpu:
            n = max(scpu) + 1
            half_jobs.append(((i, 0), half_job(tid, 0, n)))
            half_jobs.append(((i, 1), half_job(tid, 1, n)))
    halves = run_pool(half_jobs, workers, tmpdir)
    no_stmt = sorted({i for (i, _), ok in halves.items() if not ok})
    C.log(f"statement units: {len(half_jobs) // 2} tests checked, {len(no_stmt)} have dependent statements, "
          f"{time.time() - t_start:.1f}s")

    slow = [i for i, t in enumerate(tests) if t["seconds"] >= 0.5]
    cpus = run_pool([(i, cpu_job(ids[i])) for i in slow], workers, tmpdir)
    C.log(f"timed {len(cpus)} slow tests, {time.time() - t_start:.1f}s")

    # Assemble.
    files = {}
    def fidx(path):
        rel = path[len(ROOT_S):]
        i = files.get(rel)
        if i is None:
            i = files[rel] = len(files)
        return i

    sets = {}
    def set_id(members):
        b = array("H", sorted(members)).tobytes()
        i = sets.get(b)
        if i is None:
            i = sets[b] = len(sets)
        return i

    by_line = {}
    by_class = {}
    by_open = {}
    base_ok = []
    v = []
    for i, tid in enumerate(ids):
        tr = traces.get(i)
        if tr is None:
            C.log("no trace for", tid)
            base_ok.append(False)
            v.append(tests[i]["seconds"])
            continue
        ok, err = tr["result"]
        if not ok:
            C.log("baseline failure", tid, (err or "")[-200:])
        base_ok.append(bool(ok))
        at_import = modlines.get(tid.split("::")[0]) or ()
        tr["lines"].update(at_import)
        if tr.get("calls"):
            tr["calls"]["rest_lines"].update(at_import)     # seen outside every unit of the test
        for fn, line in tr["lines"]:
            by_line.setdefault((fidx(fn), line), []).append(i)
        for k in tr["classes"]:
            by_class.setdefault(k, []).append(i)
        for p in tr["opened"]:
            by_open.setdefault(p, []).append(i)
        c = cpus.get(i)
        v.append(c[1] if c else tests[i]["seconds"] * 0.8)

    # Pool calls: which call runs which line, what it returned, and its share of the test's CPU.
    fut_dir = os.path.join(out_dir, "fut")
    os.makedirs(fut_dir, exist_ok=True)
    unit_base = {}
    call_frac = {}
    masks = {}
    masks2 = {}
    masks3 = {}
    gen_check = {}
    n_gens = 0
    cmasks = {}
    n_units = 0
    n_stmt_units = 0
    n_staged = 0
    payload_bytes = 0
    for i, tid in enumerate(ids):
        tr = traces.get(i)
        calls = tr and tr.get("calls")
        if not calls or not base_ok[i]:
            continue
        rec = calls["record"]
        scpu = calls["stmt_cpu"]
        n = (max(rec) + 1) if rec else 0           # pool calls come first in the test's unit range
        ns = (max(scpu) + 1) if scpu else 0        # then its statement units
        unit_base[i] = (n_units, n, ns)             # the bit after the last unit stands for the rest of the test
        rest_bit = 1 << (n_units + n + ns)
        for fn, line in calls["rest_lines"]:
            key = (fidx(fn), line)
            masks[key] = masks.get(key, 0) | rest_bit
        n_stmt_units += ns
        total = max(tr["cpu"], 1e-6)
        call_frac[i] = [min(1.0, rec[k][2] / total) if k in rec else 0.0 for k in range(n)] + \
                       [min(1.0, scpu.get(k, 0.0) / total) for k in range(ns)]
        if rec:
            with open(os.path.join(fut_dir, f"{i}.pkl"), "wb") as f:
                gens = calls.get("gens") or {}
                pickle.dump({k: (r[0], r[1], r[3], gens.get(k)) for k, r in rec.items()}, f, protocol=5)
                n_gens += len(gens)
            payload_bytes += sum(len(r[1]) for r in rec.values() if r[1])
            n_staged += sum(1 for r in rec.values() if r[3] is not None)
        for (fn, line), m in calls["lines"].items():
            key = (fidx(fn), line)
            masks[key] = masks.get(key, 0) | (m << n_units)
        for (fn, line), m in calls["lines2"].items():
            key = (fidx(fn), line)
            masks2[key] = masks2.get(key, 0) | (m << n_units)
        for (fn, line), m in calls.get("lines3", {}).items():
            key = (fidx(fn), line)
            masks3[key] = masks3.get(key, 0) | (m << n_units)
        for fn, line in calls.get("gen_check", ()):
            gen_check.setdefault(fidx(fn), set()).add(line)
        for (fn, line), m in calls["stmt_lines"].items():
            key = (fidx(fn), line)
            masks[key] = masks.get(key, 0) | (m << (n_units + n))
        every_call = (1 << n) - 1
        every_stmt = ((1 << ns) - 1) << n
        for key, m in calls["classes"].items():
            cmasks[key] = cmasks.get(key, 0) | (m << n_units)
        for key, m in calls["stmt_classes"].items():
            cmasks[key] = cmasks.get(key, 0) | (m << (n_units + n))
        for key in calls["setup_classes"]:
            cmasks[key] = cmasks.get(key, 0) | ((every_call | every_stmt) << n_units) | rest_bit
        for key in calls["rest_classes"]:
            cmasks[key] = cmasks.get(key, 0) | (every_stmt << n_units) | rest_bit
        n_units += n + ns + 1
    nbytes = (n_units + 7) // 8
    bitsets = {}
    ulines = {}
    for (f, line), m in masks.items():
        b = m.to_bytes(nbytes, "little")
        j = bitsets.get(b)
        if j is None:
            j = bitsets[b] = len(bitsets)
        ulines.setdefault(f, {})[line] = j
    ulines2 = {}
    for (f, line), m in masks2.items():
        b = m.to_bytes(nbytes, "little")
        j = bitsets.get(b)
        if j is None:
            j = bitsets[b] = len(bitsets)
        ulines2.setdefault(f, {})[line] = j
    ulines3 = {}
    for (f, line), m in masks3.items():
        b = m.to_bytes(nbytes, "little")
        j = bitsets.get(b)
        if j is None:
            j = bitsets[b] = len(bitsets)
        ulines3.setdefault(f, {})[line] = j
    uclasses = {}
    for key, m in cmasks.items():
        b = m.to_bytes(nbytes, "little")
        j = bitsets.get(b)
        if j is None:
            j = bitsets[b] = len(bitsets)
        uclasses[key] = j
    bit_list = [None] * len(bitsets)
    for b, j in bitsets.items():
        bit_list[j] = b
    C.log(f"units: {n_units - n_stmt_units - len(unit_base)} pool calls ({n_staged} staged) and {n_stmt_units} statements "
          f"in {len(unit_base)} tests, "
          f"{payload_bytes // 1024} KiB of results, {len(bit_list)} distinct unit sets, "
          f"{len(no_stmt)} tests with dependent statements, {n_gens} calls with replayable generator calls")

    lines = {}
    for (f, line), members in by_line.items():
        lines.setdefault(f, {})[line] = set_id(members)
    imp = {}
    for fn, line in import_lines:
        imp.setdefault(fidx(fn), set()).add(line)
    setup_info = {}
    for k, tr in setups.items():
        if tr is None:
            continue
        sl = {}
        for fn, line in tr["lines"]:
            sl.setdefault(fidx(fn), set()).add(line)
        setup_info[k] = dict(cpu=tr["cpu"], lines=sl, opened=sorted(tr["opened"]), state=tr["result"])

    # Table reads: one file per table, so a run loads only the tables a patch changed.
    key_dir = os.path.join(out_dir, "keys")
    os.makedirs(key_dir, exist_ok=True)
    for name in os.listdir(key_dir):         # table ids are only meaningful within one build
        os.unlink(os.path.join(key_dir, name))
    key_untracked = []           # tests whose reads are not known: they count as reading everything
    by_table = {}
    n_reads = 0
    for i in range(len(ids)):
        r = key_res.get(i)
        if not r or not r[0] or not base_ok[i]:
            key_untracked.append(i)
            continue
        for t in r[1]["whole"]:
            by_table.setdefault(t, [[], [], {}])[0].append(i)
        for t in r[1]["len"]:
            by_table.setdefault(t, [[], [], {}])[1].append(i)
        for t, hs in r[1]["keys"].items():
            slot = by_table.setdefault(t, [[], [], {}])[2]
            n_reads += len(hs)
            for h in hs:
                a = slot.get(h)
                if a is None:
                    a = slot[h] = array("H")
                a.append(i)
    for t, (whole, size, keys) in by_table.items():
        with open(os.path.join(key_dir, f"{t}.pkl"), "wb") as f:
            pickle.dump((array("H", whole).tobytes(), array("H", size).tobytes(),
                         {h: a.tobytes() for h, a in keys.items()}), f, protocol=5)
    if not key_res:
        key_table = {}
    C.log(f"table reads: {len(by_table)} tables read by tests, {n_reads} (test, table, key) reads, "
          f"{len(key_untracked)} tests without them")

    class_sets = {k: set_id(m) for k, m in by_class.items()}
    open_sets = {k: set_id(m) for k, m in by_open.items()}
    set_list = [None] * len(sets)
    for b, i in sets.items():
        set_list[i] = b
    maps = dict(
        tests=ids, files=files, sets=set_list, lines=lines, import_lines=imp,
        import_funcs=import_funcs, unstable=unstable, lazy=lazy,
        unit_base=unit_base, call_frac=call_frac, ulines=ulines, ulines2=ulines2, ulines3=ulines3, gen_check=gen_check, uclasses=uclasses, ubits=bit_list,
        no_stmt=no_stmt,
        classes=class_sets, opened=open_sets,
        setup=setup_info, v=v, base_ok=base_ok, core_modules=core,
        test_importers=test_import_graph(),
        key_table=key_table, key_untracked=key_untracked, key_read=sorted(by_table),
    )
    with open(os.path.join(out_dir, "maps.pkl"), "wb") as f:
        pickle.dump(maps, f, protocol=5)
    C.log(f"maps: {len(files)} files, {len(by_line)} lines, {len(set_list)} sets, "
          f"{sum(len(b) for b in set_list) // 1024} KiB of sets, {sum(not x for x in base_ok)} baseline failures, "
          f"{time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
