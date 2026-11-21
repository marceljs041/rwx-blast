"""Build time: run every test of the pinned commit once, in a fork of the warm process, and record what it
touches. Everything written here describes the pinned commit only: lines executed, classes instantiated, files
opened, CPU used, and DuckDB answers to the fixture queries."""
import ast
import json
import os
import pickle
import subprocess
import sys
import time
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from br import common as C
from br import stmts

ROOT_S = C.ROOT + "/"
mon = sys.monitoring
TID = 3


def start_line_trace(sink, funcs=None):
    def on_line(code, line):
        fn = code.co_filename
        if fn.startswith(ROOT_S):
            sink.add((fn, line))
            if funcs is not None and code.co_flags & 1:
                funcs.add((fn[len(ROOT_S):], code.co_firstlineno, code.co_name))
        return mon.DISABLE
    mon.use_tool_id(TID, "br")
    mon.register_callback(TID, mon.events.LINE, on_line)
    mon.set_events(TID, mon.events.LINE)


def stop_trace():
    mon.set_events(TID, 0)
    mon.register_callback(TID, mon.events.LINE, None)
    mon.free_tool_id(TID)


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
    state = {"sink": lines, "cur": None, "types": None, "t0": 0.0}
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
            trace_into(state["cur2"])
            mon.restart_events()
            return
        if phase == 0:
            state["cur"] = set()
            state["cur2"] = None
            state["types"] = call_types.setdefault(k, set())
            trace_into(state["cur"])
        else:
            state["types"] = None
            if state["cur2"] is not None:
                bit2 = 1 << k
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

    def trace_into(sink):
        def on_line(code, line):
            fn_ = code.co_filename
            if fn_.startswith(ROOT_S):
                sink.add((fn_, line))
            return mon.DISABLE
        mon.register_callback(TID, mon.events.LINE, on_line)

    sys.addaudithook(hook)
    start_line_trace(lines)
    mon.restart_events()   # lines switched off while tracing the imports must report again
    if per_call:
        C.FC.hook = on_call
        C.FC.record = {}
        stmts.SU.hook = on_unit
    mon.register_callback(TID, mon.events.PY_START, on_start)
    for code in inits:
        mon.set_local_events(TID, code, mon.events.PY_START)
    t0 = time.process_time()
    result = fn()
    cpu = time.process_time() - t0
    mon.set_events(TID, 0)
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
        calls = dict(record=C.FC.record, lines=call_lines, lines2=call_lines2, classes=class_masks(call_types),
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
    start_line_trace(import_lines, import_funcs)
    C.warm()
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

    def record_memo():
        memo = {}
        R.after_setup = lambda c: C.wrap_executor(c, memo, record=True)
        res = [R.run(i) for i in memo_tests]
        return memo, res

    memo, memo_res = run_pool([("memo", record_memo)], 1, tmpdir)["memo"]
    C.log(f"duck memo: {sum(len(m) for m in memo.values())} queries, "
          f"{sum(not r[0] for r in memo_res)} of {len(memo_res)} tests failed while recording, "
          f"{time.time() - t_start:.1f}s")
    memo = {sec: {sql: slot for sql, slot in m.items() if slot} for sec, m in memo.items()}
    with open(os.path.join(out_dir, "duck.pkl"), "wb") as f:
        pickle.dump(memo, f, protocol=5)
    R.after_setup = lambda c: C.wrap_executor(c, memo)

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
                pickle.dump({k: (r[0], r[1], r[3]) for k, r in rec.items()}, f, protocol=5)
            payload_bytes += sum(len(r[1]) for r in rec.values() if r[1])
            n_staged += sum(1 for r in rec.values() if r[3] is not None)
        for (fn, line), m in calls["lines"].items():
            key = (fidx(fn), line)
            masks[key] = masks.get(key, 0) | (m << n_units)
        for (fn, line), m in calls["lines2"].items():
            key = (fidx(fn), line)
            masks2[key] = masks2.get(key, 0) | (m << n_units)
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
          f"{len(no_stmt)} tests with dependent statements")

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

    class_sets = {k: set_id(m) for k, m in by_class.items()}
    open_sets = {k: set_id(m) for k, m in by_open.items()}
    set_list = [None] * len(sets)
    for b, i in sets.items():
        set_list[i] = b
    maps = dict(
        tests=ids, files=files, sets=set_list, lines=lines, import_lines=imp,
        import_funcs=import_funcs, unstable=unstable, lazy=lazy,
        unit_base=unit_base, call_frac=call_frac, ulines=ulines, ulines2=ulines2, uclasses=uclasses, ubits=bit_list,
        no_stmt=no_stmt,
        classes=class_sets, opened=open_sets,
        setup=setup_info, v=v, base_ok=base_ok, core_modules=core,
        test_importers=test_import_graph(),
    )
    with open(os.path.join(out_dir, "maps.pkl"), "wb") as f:
        pickle.dump(maps, f, protocol=5)
    C.log(f"maps: {len(files)} files, {len(by_line)} lines, {len(set_list)} sets, "
          f"{sum(len(b) for b in set_list) // 1024} KiB of sets, {sum(not x for x in base_ok)} baseline failures, "
          f"{time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
