"""Shared pieces: patch parsing, overlay trees, inline process pool, test runner, code diff, packing."""
import concurrent.futures as cf
import hashlib
import importlib
import marshal
import os
import pickle
import re
import signal
import sys
import time
import types
import unittest

from br import stmts

ROOT = os.environ.get("BR_ROOT", "/opt/sqlglot").rstrip("/")
MAPS = os.environ.get("BR_MAPS", "/opt/br/maps")
DATA = os.environ.get("BR_DATA", "/data")
TMP = os.environ.get("BR_TMP", "/tmp")


def log(*a):
    print(*a, file=sys.stderr, flush=True)


# ---------------------------------------------------------------- patches

class PatchError(Exception):
    pass


_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


def parse_patch(text):
    """Returns a list of file records: path, old_path, new, deleted, binary, hunks[(old_start, [(tag, line)])]."""
    files = []
    cur = None
    lines = text.split("\n")
    i = 0
    n = len(lines)
    while i < n:
        l = lines[i]
        if l.startswith("diff --git "):
            m = re.match(r"diff --git a/(.*) b/(.*)$", l)
            if not m:
                raise PatchError("header")
            cur = dict(path=m.group(2), old_path=m.group(1), new=False, deleted=False, binary=False, hunks=[])
            files.append(cur)
        elif cur is None:
            pass
        elif l.startswith("new file mode"):
            cur["new"] = True
        elif l.startswith("deleted file mode"):
            cur["deleted"] = True
        elif l.startswith("rename from "):
            cur["old_path"] = l[12:]
        elif l.startswith("rename to "):
            cur["path"] = l[10:]
        elif l.startswith("Binary files ") or l.startswith("GIT binary patch"):
            cur["binary"] = True
        elif l.startswith("@@"):
            m = _HUNK.match(l)
            if not m:
                raise PatchError("hunk")
            want_old = int(m.group(2) or 1)
            want_new = int(m.group(4) or 1)
            body = []
            i += 1
            co = cn = 0
            while i < n and (co < want_old or cn < want_new):
                b = lines[i]
                if b.startswith("\\"):
                    body.append(("\\", ""))
                else:
                    tag = b[:1] or " "
                    if tag not in " +-":
                        raise PatchError("hunk body")
                    body.append((tag, b[1:]))
                    co += tag != "+"
                    cn += tag != "-"
                i += 1
            if i < n and lines[i].startswith("\\"):
                body.append(("\\", ""))
                i += 1
            cur["hunks"].append((int(m.group(1)), body))
            continue
        i += 1
    return files


def apply_file(old, rec):
    """Applies one file record to its old text. Returns (new_text, changed_old_lines)."""
    old_nl = old.endswith("\n")
    ol = old.split("\n")
    if old_nl:
        ol.pop()
    if old == "":
        ol = []
    new = []
    pos = 0
    changed = set()
    new_nl = old_nl or old == ""
    for start, body in rec["hunks"]:
        s = max(start - 1, 0)
        if s < pos or s > len(ol):
            raise PatchError("hunk position")
        new.extend(ol[pos:s])
        pos = s
        has_minus = any(t == "-" for t, _ in body)
        last = " "
        for tag, txt in body:
            if tag == "\\":
                if last == "+":
                    new_nl = False
                elif last == "-":
                    new_nl = True
                else:
                    new_nl = False
                continue
            last = tag
            if tag == "+":
                new.append(txt)
                if not has_minus:
                    changed.add(pos)      # line before the insertion (1-based: pos)
                    changed.add(pos + 1)  # line after the insertion
                continue
            if pos >= len(ol) or ol[pos] != txt:
                raise PatchError("context mismatch")
            if tag == " ":
                new.append(txt)
            else:
                changed.add(pos + 1)
            pos += 1
    new.extend(ol[pos:])
    text = "\n".join(new)
    if new and new_nl:
        text += "\n"
    changed.discard(0)
    return text, sorted(changed)


def build_overlay(root, ov, changes):
    """Makes a tree at `ov` that mirrors `root` through symlinks, except for `changes` {rel: bytes or None}."""
    real = {""}
    for rel in changes:
        parts = rel.split("/")
        for i in range(1, len(parts)):
            real.add("/".join(parts[:i]))
    for d in sorted(real, key=len):
        dst = os.path.join(ov, d) if d else ov
        os.makedirs(dst, exist_ok=True)
        src = os.path.join(root, d) if d else root
        if not os.path.isdir(src):
            continue
        for name in os.listdir(src):
            rel = f"{d}/{name}" if d else name
            if rel in real or rel in changes or name == ".git":
                continue
            os.symlink(os.path.join(src, name), os.path.join(ov, rel))
    for rel, data in changes.items():
        if data is not None:
            with open(os.path.join(ov, rel), "wb") as f:
                f.write(data)


# ---------------------------------------------------------------- inline pool

class Timeout(BaseException):
    pass


class FutureCtx:
    """Which test is running, and what is known about the pool calls it makes on the pinned commit."""
    test = None       # token of the running test; calls made outside a test are never reused
    count = 0         # calls submitted so far by this test
    reuse = None      # {k: (input_hash, pickled_result)} recorded on the pinned commit
    affected = None   # calls that execute a changed statement and must really run
    hook = None       # build time: called with (0|1, k) around each call
    record = None     # build time: {k: (input_hash, pickled_result | None, cpu, stage_hash | None)}
    probe = None      # calls whose later stage runs no changed statement: stop after the first stage if its
                      # output is byte-identical to the pinned commit's
    cur = None        # the call that is running right now


FC = FutureCtx()


def input_hash(fn, payload, pool=None):
    """Everything a pool call is given: its function, its pickled arguments, and the pickled arguments of the
    worker initializer of its pool (the state a spawned worker would start from)."""
    h = hashlib.blake2b(payload, digest_size=12)
    h.update(getattr(fn, "__module__", "").encode())
    h.update(getattr(fn, "__qualname__", "").encode())
    if pool is not None:
        h.update(pool._init_hash)
    return h.digest()


class _Reuse(BaseException):
    """Raised inside a pool call when its first stage reproduced the pinned commit's output exactly."""


def stage_done(value):
    """Called when the first stage of a staged pool call (the optimizer run inside `execute`) returns."""
    fut = FC.cur
    if fut is None or fut._staged:
        return
    fut._staged = True
    if FC.record is None and fut._want is None:
        return
    try:
        h = hashlib.blake2b(pickle.dumps(value), digest_size=12).digest()
    except Exception:
        return
    fut._stage_hash = h
    if FC.record is not None:
        if FC.hook:
            FC.hook(4, fut._k)            # build time: lines from here on belong to the later stage
    elif h == fut._want:
        raise _Reuse()


class LazyFuture:
    """A future that runs its call on first use, with arguments and result pickled like a spawned worker.

    When the same call, with byte-identical pickled arguments, was made on the pinned commit and executes no
    changed statement, its recorded result is unpickled instead of being computed again."""

    def __init__(self, fn, args, kwargs, pool=None):
        self._fn = fn
        self._exc = None
        self._res = None
        self._state = 0
        self._pool = pool
        self._test = FC.test
        self._k = FC.count
        self._staged = False
        self._stage_hash = None
        self._want = None
        FC.count += 1
        try:
            self._payload = pickle.dumps((args, kwargs))
        except Exception as e:  # the real pool reports this through the future
            self._exc = e
            self._state = 2

    def _run(self):
        if self._state:
            return
        self._state = 1
        k = self._k
        mine = self._test is not None and self._test is FC.test
        in_unit = stmts.SU.active
        ent = None
        if mine and not in_unit and FC.reuse is not None and FC.affected is not None:
            ent = FC.reuse.get(k)
            if ent is not None and (ent[1] is None or ent[0] != input_hash(self._fn, self._payload, self._pool)):
                ent = None
        if ent is not None and k not in FC.affected:
            self._load(ent[1])
            return
        if ent is not None and FC.probe is not None and k in FC.probe and len(ent) > 2 and ent[2] is not None:
            self._want = ent[2]
        if self._pool is not None:
            self._pool._ensure_init()
        hook = FC.hook if (mine and not in_unit) else None
        if hook:
            hook(0, k)
        t0 = time.process_time()
        out = None
        prev, FC.cur = FC.cur, (self if (mine and not in_unit) else None)
        try:
            a, kw = pickle.loads(self._payload)
            out = pickle.dumps(self._fn(*a, **kw))
            self._res = pickle.loads(out)
        except Timeout:
            FC.cur = prev
            raise
        except _Reuse:
            FC.cur = prev
            self._load(ent[1])
            return
        except BaseException as e:
            self._exc = e
            out = None
        FC.cur = prev
        if hook:
            hook(1, k)
        if mine and FC.record is not None:
            FC.record[k] = (input_hash(self._fn, self._payload, self._pool), None if in_unit else out,
                            0.0 if in_unit else time.process_time() - t0, None if in_unit else self._stage_hash)
        self._state = 2

    def _load(self, payload):
        try:
            self._res = pickle.loads(payload)
        except Timeout:
            raise
        except BaseException as e:
            self._exc = e
        self._state = 2

    def result(self, timeout=None):
        self._run()
        if self._exc is not None:
            raise self._exc
        return self._res

    def exception(self, timeout=None):
        self._run()
        return self._exc

    def done(self):
        return True

    def cancel(self):
        return False


class InlinePool:
    """Runs submitted calls in this process, on first use of their result. The worker initializer runs before
    the first call that really executes, so a test whose calls all reuse recorded results never pays for it."""

    def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **kw):
        self._init = None
        self._init_hash = b""
        if initializer is not None:
            payload = pickle.dumps(initargs)
            self._init = (initializer, payload)
            self._init_hash = hashlib.blake2b(payload, digest_size=12).digest()

    def _ensure_init(self):
        if self._init is None:
            return
        (initializer, payload), self._init = self._init, None
        if FC.hook:
            FC.hook(2, -1)
        initializer(*pickle.loads(payload))
        if FC.hook:
            FC.hook(3, -1)

    def submit(self, fn, /, *args, **kwargs):
        return LazyFuture(fn, args, kwargs, self)

    def map(self, fn, *iterables, **kw):
        self._ensure_init()
        return [fn(*pickle.loads(pickle.dumps(a))) for a in zip(*iterables)]

    def shutdown(self, wait=True, cancel_futures=False):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


_real_wait = cf.wait
_real_as_completed = cf.as_completed


def _wait(fs, timeout=None, return_when=cf.ALL_COMPLETED):
    fs = list(fs)
    if fs and all(isinstance(f, LazyFuture) for f in fs):
        return set(fs), set()
    return _real_wait(fs, timeout, return_when)


def _as_completed(fs, timeout=None):
    fs = list(fs)
    if fs and all(isinstance(f, LazyFuture) for f in fs):
        return iter(fs)
    return _real_as_completed(fs, timeout)


def install_pool():
    import concurrent.futures.process as p
    cf.ProcessPoolExecutor = InlinePool
    p.ProcessPoolExecutor = InlinePool
    cf.wait = _wait
    cf.as_completed = _as_completed


# ---------------------------------------------------------------- test runner

def split_id(tid):
    f, cls, meth = tid.split("::")
    return f[:-3].replace("/", "."), cls, meth


class Runner:
    def __init__(self):
        self.classes = {}
        self.setup = {}  # class -> None (ok) | "skip" | error text
        self.after_setup = None  # hook(cls)
        self.instrument = True   # rewrite test methods into statement units on import

    def get_class(self, tid):
        mod, cls, meth = split_id(tid)
        key = (mod, cls)
        c = self.classes.get(key)
        if c is None:
            m = importlib.import_module(mod)
            patch_duckdb()
            if self.instrument:
                stmts.instrument(m)
            c = getattr(m, cls)
            self.classes[key] = c
        return c, meth

    def ensure_setup(self, c):
        if c in self.setup:
            return
        try:
            c.setUpClass()
            self.setup[c] = None
            if self.after_setup:
                self.after_setup(c)
        except unittest.SkipTest:
            self.setup[c] = "skip"
        except Timeout:
            raise
        except BaseException as e:
            self.setup[c] = "setUpClass: " + repr(e)[:200]

    def prepare(self, tid):
        """Imports the test's module and runs its class set-up, so that neither counts against the test's
        own time limit."""
        try:
            c, _ = self.get_class(tid)
        except Timeout:
            raise
        except BaseException:
            return
        self.ensure_setup(c)

    def run(self, tid):
        """Returns (ok, error_text)."""
        try:
            c, meth = self.get_class(tid)
        except Timeout:
            raise
        except BaseException as e:
            return False, "import: " + repr(e)[:200]
        self.ensure_setup(c)
        st = self.setup[c]
        if st == "skip":
            return True, None
        if st is not None:
            return False, st
        try:
            t = c(meth)
        except Exception as e:
            return False, "no such test: " + repr(e)[:100]
        res = unittest.TestResult()
        res.failfast = True
        FC.test = object()
        FC.count = 0
        stmts.SU.active = False
        try:
            t.run(res)
        finally:
            FC.test = None
            stmts.SU.active = False
        if res.wasSuccessful():
            return True, None
        bad = res.failures + res.errors
        return False, (bad[0][1][-400:] if bad else "unexpected success")


def _on_alarm(signum, frame):
    raise Timeout()


def arm(seconds):
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, seconds)


def disarm():
    signal.setitimer(signal.ITIMER_REAL, 0)


# ---------------------------------------------------------------- code diff

def load_old_code(path, src):
    """The compiled baseline module, from its cached pyc when that is current."""
    try:
        pyc = importlib.util.cache_from_source(path)
        with open(pyc, "rb") as f:
            data = f.read()
        st = os.stat(path)
        if data[:4] == importlib.util.MAGIC_NUMBER and data[4:8] == b"\0\0\0\0" \
                and int.from_bytes(data[8:12], "little") == int(st.st_mtime) & 0xFFFFFFFF \
                and int.from_bytes(data[12:16], "little") == st.st_size & 0xFFFFFFFF:
            return marshal.loads(data[16:])
    except Exception:
        pass
    return compile(src, path, "exec")


def sem_key(c, mask_doc=False):
    """What a code object does, without where it sits in the file. A class body stores its own first line
    number as a constant (__firstlineno__); that one is masked so a shifted class is not a changed class."""
    first = None if c.co_flags & 0x1 else c.co_firstlineno
    consts = [(type(x), None if (type(x) is int and x == first) else x)
              for x in c.co_consts if not isinstance(x, types.CodeType)]
    if mask_doc and consts:
        consts[0] = None          # the caller knows this function starts with a docstring
    consts = tuple(consts)
    return (c.co_code, c.co_names, c.co_varnames, c.co_freevars, c.co_cellvars, c.co_flags, c.co_argcount,
            c.co_kwonlyargcount, c.co_posonlyargcount, c.co_exceptiontable, consts)


def is_function(code):
    return bool(code.co_flags & 0x1)  # CO_OPTIMIZED: functions, lambdas, generator expressions


def diff_tree(o, n, path, out):
    """Collects (old, new, own_changed, path) for every pair with a difference. Raises ValueError when the
    nesting of code objects is not the same."""
    oc = [x for x in o.co_consts if isinstance(x, types.CodeType)]
    nc = [x for x in n.co_consts if isinstance(x, types.CodeType)]
    if len(oc) != len(nc):
        raise ValueError("structure")
    own = sem_key(o) != sem_key(n)
    sub = False
    for a, b in zip(oc, nc):
        if a.co_name != b.co_name:
            raise ValueError("structure")
        if diff_tree(a, b, path + [a.co_name], out):
            sub = True
    if own or sub:
        out.append((o, n, own, path))
    return own or sub


def own_lines(code):
    return {l for _, _, l in code.co_lines() if l is not None}


def widen(code, lines):
    """Lines of `code` whose instructions span any of `lines` (a continuation line has no event of its own)."""
    out = set()
    lo = min(lines)
    hi = max(lines)
    for pos in code.co_positions():
        a, b = pos[0], pos[1]
        if a is None:
            continue
        if b is None:
            b = a
        if b < lo or a > hi:
            continue
        for x in lines:
            if a <= x <= b:
                out.add(a)
                break
    return out


# ---------------------------------------------------------------- packing

WORKER_MS = 2000


def pack(numbers, ms):
    """Best split over 1..8 workers, longest first. Returns (cost_ms, workers)."""
    if not numbers:
        return 0, []
    order = sorted(numbers, key=lambda x: -ms[x])
    total = sum(ms[x] for x in order)
    best = None
    for k in range(1, 9):
        if best is not None and WORKER_MS * k + max(ms[order[0]], total // k) >= best[0]:
            continue
        loads = [0] * k
        ws = [[] for _ in range(k)]
        for x in order:
            i = loads.index(min(loads))
            loads[i] += ms[x]
            ws[i].append(x)
        cost = WORKER_MS * k + max(loads)
        if best is None or cost < best[0]:
            best = (cost, ws)
    return best[0], [w for w in best[1] if w]


def pack_cost(numbers, ms):
    """Cheap lower-bound style estimate of pack()'s cost, good enough for decisions."""
    if not numbers:
        return 0
    total = 0
    big = 0
    for x in numbers:
        m = ms[x]
        total += m
        if m > big:
            big = m
    return min(WORKER_MS * k + max(big, -(-total // k)) for k in range(1, 9))


# ---------------------------------------------------------------- warm process

def _pretend_pytest():
    """The commits were labelled under pytest, and sqlglot looks for "pytest" in sys.modules. A placeholder
    keeps that answer the same here; it turns into the real module if anything ever uses it."""
    if "pytest" in sys.modules:
        return

    class _Lazy(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith("__"):
                raise AttributeError(name)
            del sys.modules["pytest"]
            real = importlib.import_module("pytest")
            return getattr(real, name)

    sys.modules["pytest"] = _Lazy("pytest")


def warm():
    """Puts the process in the state every test child is forked from: sqlglot fully imported, pools inlined."""
    import pkgutil
    os.environ["SKIP_INTEGRATION"] = "0"
    os.environ["PYTHONPATH"] = ROOT
    _pretend_pytest()
    sys.dont_write_bytecode = True
    if ROOT not in sys.path:
        sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    install_pool()
    import sqlglot
    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if m.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(m.name)
        except Exception as e:
            log("skip module", m.name, repr(e)[:80])
    patch_stages()


# ---------------------------------------------------------------- DuckDB connections

class _Deferred:
    """Result of a statement that was only noted down. Using it runs the statement for real."""

    def __init__(self, owner, sql):
        self._owner, self._sql = owner, sql

    def __getattr__(self, name):
        owner = self._owner
        if owner._conn is None and self._sql in owner._log:
            owner._log.remove(self._sql)
        return getattr(owner._real().execute(self._sql), name)


class LazyConn:
    """A DuckDB connection that is opened on first real use.

    DuckDB starts one worker thread per host core for every database, and worker threads do not survive a
    fork. Statements whose result nobody looks at (the CREATE and INSERT statements of a class set-up) are
    noted down and replayed by whichever process first needs an answer, so the threads are created in the
    child that uses them, and not at all when every answer comes from the cache below."""

    defer = True

    def __init__(self, connect, args, kwargs):
        self._mk = (connect, args, kwargs)
        self._conn = None
        self._log = []

    def _real(self):
        if self._conn is None:
            connect, a, k = self._mk
            self._conn = connect(*a, **k)
            log, self._log = self._log, []
            for sql in log:
                self._conn.execute(sql)
        return self._conn

    def execute(self, sql, *a, **k):
        if self._conn is None and LazyConn.defer and not a and not k and isinstance(sql, str):
            self._log.append(sql)
            return _Deferred(self, sql)
        return self._real().execute(sql, *a, **k)

    def close(self):
        if self._conn is not None:
            self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def __getattr__(self, name):
        return getattr(self._real(), name)


def patch_stages():
    """Marks the end of the first stage of `sqlglot.executor.execute`: its optimizer run. What follows
    (planning and running the plan) depends on the code only through that stage's output, plus the lines the
    later stage itself executes, which are recorded separately."""
    ex = sys.modules.get("sqlglot.executor")
    real = getattr(ex, "optimize", None) if ex else None
    if real is None or getattr(real, "__br_stage__", False):
        return

    def optimize(*a, **k):
        out = real(*a, **k)
        if FC.cur is not None:
            stage_done(out)
        return out

    optimize.__br_stage__ = True
    optimize.__wrapped__ = real
    ex.optimize = optimize


def patch_duckdb():
    """Caps DuckDB at two threads per database (the run has two cores and a limit of 256 threads)."""
    d = sys.modules.get("duckdb")
    if d is None or getattr(d, "__br_patched__", False):
        return
    real = d.connect

    def connect(*a, **k):
        cfg = dict(k.get("config") or {})
        cfg.setdefault("threads", 2)
        k["config"] = cfg
        return LazyConn(real, a, k)

    d.connect = connect
    d.__br_patched__ = True


# ---------------------------------------------------------------- DuckDB reference cache

class _DuckResult:
    """The result of a query whose answers on the pinned commit are (or are being) recorded in `slot`."""

    def __init__(self, conn, sql, slot, live=None):
        self._conn, self._sql, self._slot, self._live = conn, sql, slot, live

    def _result(self):
        live, self._live = self._live, None
        return live if live is not None else self._conn.execute(self._sql)

    def fetchdf(self):
        if "df" not in self._slot:
            self._slot["df"] = self._result().fetchdf()
        return self._slot["df"].copy()

    df = fetchdf

    def fetchall(self):
        if "rows" not in self._slot:
            self._slot["rows"] = self._result().fetchall()
        return list(self._slot["rows"])

    def __getattr__(self, name):
        return getattr(self._result(), name)


class DuckProxy:
    """Wraps a DuckDB connection: a query whose exact text was answered on the pinned commit is not run again.
    Statements nobody fetched from (CREATE, INSERT) have no recorded answer and always run."""

    def __init__(self, conn, memo, record=False):
        self._conn, self._memo, self._record = conn, memo, record

    def execute(self, sql, *a, **k):
        if a or k or not isinstance(sql, str):
            return self._conn.execute(sql, *a, **k)
        if self._record:
            return _DuckResult(self._conn, sql, self._memo.setdefault(sql, {}), self._conn.execute(sql))
        slot = self._memo.get(sql)
        if slot:
            return _DuckResult(self._conn, sql, slot)
        return self._conn.execute(sql)

    def __getattr__(self, name):
        return getattr(self._conn, name)


MEMO_CONNS = {
    ("tests.test_executor", "TestExecutor"): ("tpch_conn", "tpcds_conn"),
    ("tests.test_optimizer", "TestOptimizer"): ("conn",),
}


def wrap_executor(cls, memo, record=False):
    """Puts the answer cache in front of the DuckDB connections a test class opens in its set-up. Each class
    gets its own section of the cache, because the same query text means different tables in each."""
    names = MEMO_CONNS.get((cls.__module__, cls.__name__))
    if not names:
        return
    for name in names:
        conn = getattr(cls, name, None)
        if conn is not None and not isinstance(conn, DuckProxy):
            setattr(cls, name, DuckProxy(conn, memo.setdefault(f"{cls.__name__}.{name}", {}), record))


def reimport(ov):
    """Drops every sqlglot and tests module and imports sqlglot again from the tree at `ov`."""
    import pkgutil
    for name in [n for n in sys.modules if n.split(".")[0] in ("sqlglot", "tests")]:
        del sys.modules[name]
    sys.path[:] = [ov if p == ROOT else p for p in sys.path]
    os.chdir(ov)
    os.environ["PYTHONPATH"] = ov
    importlib.invalidate_caches()
    import sqlglot
    failed = []
    for m in pkgutil.walk_packages(sqlglot.__path__, "sqlglot."):
        if m.name.endswith("__main__"):
            continue
        try:
            importlib.import_module(m.name)
        except Exception as e:
            failed.append(m.name)
            log("import failed", m.name, repr(e)[:120])
    patch_stages()
    return failed
