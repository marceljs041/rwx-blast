"""Run time: for every patch, decide which tests can fail by running the affected ones against the patched
code in a fork of a warm process, then print the selection split over workers."""
import ast
import gc
import importlib
import json
import os
import pickle
import re
import resource
import select
import shutil
import sys
import time
import types
from array import array

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from br import common as C
from br import statediff
from br import stmts

T0 = time.time()
PYTEST_CONFIG = re.compile(r"^\[(tool\.pytest|pytest|tool:pytest)", re.M)
CONFIG_FILES = {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"}
HEAVY_MS = 3000
COSTLY_S = 3.0
# The only test the skip rule was measured for.
ASSUME_PASS = ("::TestExecutor::test_execute_tpcds",)


class State:
    pass


class NeedReimport(Exception):
    pass


S = State()


def load():
    S.tests = json.load(open(os.path.join(C.DATA, "tests.json")))["tests"]
    S.ms = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in S.tests}
    S.id_of = {t["number"]: t["id"] for t in S.tests}
    S.all = sorted(S.ms)
    num_of = {t["id"]: t["number"] for t in S.tests}
    with open(os.path.join(C.MAPS, "maps.pkl"), "rb") as f:
        m = pickle.load(f)
    S.maps = m
    S.idx2num = [num_of.get(i) for i in m["tests"]]
    known = {n for n in S.idx2num if n is not None}
    S.unknown = [n for n in S.all if n not in known]          # tests the maps never saw: always suspects
    S.v = {}
    S.base_ok = {}
    for i, n in enumerate(S.idx2num):
        if n is not None:
            S.v[n] = max(m["v"][i], 0.0005)
            S.base_ok[n] = m["base_ok"][i]
    for n in S.unknown:
        S.v[n] = S.ms[n] / 1000
        S.base_ok[n] = True
    S.num2idx = {n: i for i, n in enumerate(S.idx2num) if n is not None}
    S.bit_cache = {}
    S.by_file = {}
    for t in S.tests:
        S.by_file.setdefault(t["file"], []).append(t["number"])
    S.set_cache = {}
    S.lazy_load = [t["number"] for t in S.tests if t["id"].endswith("::test_lazy_load")]
    S.expensive = {k for k, s in m["setup"].items() if s["cpu"] >= 0.3}
    S.memo = None
    S.fidx = None
    S.fp0 = None
    S.warm = False
    S.runner = None
    S.prepared = set()


def members(set_id):
    r = S.set_cache.get(set_id)
    if r is None:
        r = S.set_cache[set_id] = frozenset(
            S.idx2num[i] for i in array("H", S.maps["sets"][set_id]) if S.idx2num[i] is not None)
    return r


def tests_for_lines(rel, lines):
    f = S.maps["files"].get(rel)
    if f is None:
        return set()
    lm = S.maps["lines"].get(f, {})
    out = set()
    for l in lines:
        s = lm.get(l)
        if s is not None:
            out |= members(s)
    return out


def _bits(j):
    b = S.bit_cache.get(j)
    if b is None:
        b = S.bit_cache[j] = int.from_bytes(S.maps["ubits"][j], "little")
    return b


def line_mask(lines_by_file):
    mask = 0
    for rel, lines in lines_by_file.items():
        ul = S.maps["ulines"].get(S.maps["files"].get(rel), {})
        for l in lines:
            j = ul.get(l)
            if j is not None:
                mask |= _bits(j)
    return mask


def state_mask(changed_keys):
    """Pool calls that can see a changed class attribute or module global."""
    mask = 0
    for kind, owner, name in changed_keys:
        if kind == "d":
            where = function_lines((kind, owner, name))
            if where is not None:
                mask |= line_mask({where[0]: where[1]})
                continue
            kind = "c" if ":" in owner else "g"
        if kind == "c" and S.maps["classes"].get(owner) is not None:
            j = S.maps["uclasses"].get(owner)
            if j is not None:
                mask |= _bits(j)
        else:
            mask |= line_mask(ref_lines(name) if not name.startswith("__") else {})
    return mask


def affected_calls(plan, cands=None, extra_mask=0, with_stmts=False):
    """{test number: set of pool-call indexes that must really run}, for tests that make pool calls."""
    mask = line_mask(plan.lines) | extra_mask
    out = {}
    for n in (plan.cands if cands is None else cands):
        i = S.num2idx.get(n)
        ub = S.maps["unit_base"].get(i)
        if ub is None:
            continue
        base, nc, ns = ub
        part = (mask >> base) & ((1 << (nc + ns)) - 1)
        calls = {k for k in range(nc) if (part >> k) & 1}
        units = None
        if with_stmts and ns and i not in S.no_stmt:
            units = {k for k in range(ns) if (part >> (nc + k)) & 1}
        out[n] = (calls, units)
    return out


def openers(rel):
    out = set()
    s = S.maps["opened"].get(rel)
    if s is not None:
        out |= members(s)
    return out


def module_name(rel):
    m = rel[:-3].replace("/", ".")
    return m[:-9] if m.endswith(".__init__") else m


def tests_importing(mod):
    out = set()
    for f in S.maps["test_importers"].get(mod, ()):
        out.update(S.by_file.get(f, ()))
    return out


# ---------------------------------------------------------------- analysis (no test is run here)

class Plan:
    def __init__(self, commit):
        self.commit = commit
        self.mode = "none"        # none | A | files | B | all
        self.cands = set()        # tests to verify
        self.forced = set()       # tests selected without verification
        self.swaps = []           # (rel, [(old_code, new_code)])
        self.changes = {}         # rel -> bytes | None, for the overlay tree
        self.lines = {}           # rel -> lines whose functions change (for set-up conflicts)
        self.statediff = False    # B: compare import state to find more suspects
        self.reuse_ok = True      # B: recorded pool-call results may be used for calls that cannot see the change
        self.broken = False       # B: a module no longer imports; test files are probed for import errors
        self.why = []


def enclosing_lines(root, line):
    """Own lines of the innermost code object of `root` that spans `line`."""
    best = None
    stack = [root]
    while stack:
        c = stack.pop()
        ls = C.own_lines(c)
        if c is not root and ls and min(ls) <= line <= max(ls) and (best is None or min(ls) >= min(C.own_lines(best))):
            best = c
        stack.extend(x for x in c.co_consts if isinstance(x, types.CodeType))
    return C.own_lines(best) if best is not None else set()


def statement_spans(src, doc_lines=None):
    """(first, last) line of every statement, lambda body, except clause and match case. A compound statement
    counts with its header only; its body statements are listed on their own. `doc_lines` collects the first
    line of every function that starts with a docstring."""
    spans = []
    for node in ast.walk(ast.parse(src)):
        if doc_lines is not None and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            b0 = node.body[0]
            if isinstance(b0, ast.Expr) and isinstance(b0.value, ast.Constant) and isinstance(b0.value.value, str):
                doc_lines.add(min([node.lineno] + [d.lineno for d in node.decorator_list]))
        if isinstance(node, ast.stmt):
            end = node.end_lineno
            body = getattr(node, "body", None)
            if isinstance(body, list) and body:
                end = max(node.lineno, body[0].lineno - 1)
            first = node.lineno
            for d in getattr(node, "decorator_list", ()):
                first = min(first, d.lineno)
            spans.append((first, end))
        elif isinstance(node, ast.Lambda):
            spans.append((node.body.lineno, node.body.end_lineno))
        elif isinstance(node, (ast.ExceptHandler, ast.match_case)):
            body = node.body
            first = node.lineno if isinstance(node, ast.ExceptHandler) else node.pattern.lineno
            spans.append((first, max(first, body[0].lineno - 1)))
    return spans


def statement_lines(spans, line):
    """Every line of the innermost statement around `line`: an edit inside a multi-line expression can change
    which of its lines run, so the statement is the smallest unit coverage can be trusted for."""
    best = None
    for a, b in spans:
        if a <= line <= b and (best is None or (b - a) < (best[1] - best[0])):
            best = (a, b)
    return set(range(best[0], best[1] + 1)) if best else {line}


def analyze_python(plan, rel, old, new, changed):
    """A modified .py under sqlglot/ or tests/. Returns False when nothing executable changed."""
    in_tests = rel.startswith("tests/")
    path = os.path.join(C.ROOT, rel)
    mod = module_name(rel)
    try:
        ncode = compile(new, path, "exec", dont_inherit=True)
    except (SyntaxError, ValueError):
        plan.mode = "B"
        plan.reuse_ok = False
        plan.why.append(f"{rel}: does not compile")
        if in_tests:
            plan.cands |= tests_importing(mod)
        else:
            plan.statediff = True
            plan.broken = True
            plan.cands |= tests_for_lines(rel, range(1, old.count("\n") + 2))
        return True
    ocode = C.load_old_code(path, old)
    out = []
    structure = True
    try:
        C.diff_tree(ocode, ncode, [], out)
    except ValueError:
        structure = False
    if structure and not out:
        return False
    imp = S.maps["import_funcs"]
    if in_tests:
        plan.reuse_ok = False
    if not structure:
        plan.reuse_ok = False
        lines = set()
        spans = statement_spans(old)
        for l in changed:
            lines |= enclosing_lines(ocode, l) | statement_lines(spans, l)
        plan.cands |= tests_for_lines(rel, lines | set(changed))
        plan.mode = "B"
        plan.why.append(f"{rel}: nesting changed")
        if in_tests:
            plan.cands |= tests_importing(mod)
        else:
            plan.statediff = True
        return True
    doc_funcs = set()
    spans = statement_spans(old, doc_funcs)
    own = []
    for o, n, is_own, p in out:
        if not is_own:
            continue
        if p and C.is_function(o) and o.co_firstlineno in doc_funcs and o.co_consts and n.co_consts \
                and type(o.co_consts[0]) is str and type(n.co_consts[0]) is str \
                and C.sem_key(o, True) == C.sem_key(n, True):
            continue                  # only the docstring differs; nothing in sqlglot or its tests reads one
        own.append((o, n, p))
    if not own:
        return False
    func_only = all(p and C.is_function(o) for o, n, p in own)
    lines = set()
    ran_at_import = False
    for o, n, p in own:
        if not (p and C.is_function(o)):
            continue
        ol = C.own_lines(o)
        if not ol:
            continue
        wide = set()
        for l in changed:
            if min(ol) <= l <= max(ol):
                wide |= statement_lines(spans, l)
        lines |= wide if wide & ol else ol
        if (rel, o.co_firstlineno, o.co_name) in imp:
            ran_at_import = True
    plan.cands |= tests_for_lines(rel, lines)
    plan.lines.setdefault(rel, set()).update(lines)
    if in_tests:
        plan.mode = "B"
        if not func_only:
            plan.cands |= tests_importing(mod)
        plan.why.append(f"{rel}: test code")
    elif func_only and not ran_at_import:
        pairs = [(o, n) for o, n, _, p in out if p and C.is_function(o)]
        plan.swaps.append((rel, pairs))
        if plan.mode in ("none", "files"):
            plan.mode = "A"
        plan.why.append(f"{rel}: {len(own)} function(s)")
    else:
        plan.mode = "B"
        plan.statediff = True
        plan.why.append(f"{rel}: {'runs at import' if func_only else 'class or module level'}")
    return True


def analyze(commit, text):
    plan = Plan(commit)
    try:
        recs = C.parse_patch(text)
        for rec in recs:
            rel, old_rel = rec["path"], rec["old_path"]
            base = os.path.basename(rel)
            is_py = rel.endswith(".py")
            in_code = rel.startswith("sqlglot/") or rel.startswith("tests/")
            renamed = old_rel != rel
            added = rec["new"] or not os.path.exists(os.path.join(C.ROOT, old_rel))
            plan.cands |= openers(old_rel) | openers(rel)
            if rec["binary"]:
                plan.forced |= openers(old_rel) | openers(rel)
                if is_py and in_code:
                    return force_all(plan, "binary change to code")
                plan.why.append(f"{rel}: binary")
                continue
            old = ""
            if not added:
                with open(os.path.join(C.ROOT, old_rel), "rb") as f:
                    old = f.read().decode("utf-8")
            if rec["deleted"]:
                new = None
            else:
                new, changed = C.apply_file(old, rec)
            if new == old and not renamed and not added:
                continue
            if renamed:
                plan.changes[old_rel] = None
            plan.changes[rel] = None if new is None else new.encode("utf-8")
            if added or renamed or new is None:
                plan.cands |= openers(os.path.dirname(rel) + "/") | openers(os.path.dirname(old_rel) + "/")
            if base == "conftest.py" or rel == "tests/__init__.py" or base in ("pytest.ini", "tox.ini"):
                return force_all(plan, f"{rel}: test configuration")
            if base in CONFIG_FILES and new is not None and PYTEST_CONFIG.search(new):
                return force_all(plan, f"{rel}: pytest configuration")
            if is_py and in_code:
                if added or renamed or new is None:
                    plan.reuse_ok = False
                    if rel.startswith("tests/"):
                        plan.mode = "B"
                        plan.cands |= tests_importing(module_name(old_rel)) | set(S.by_file.get(old_rel, ()))
                        plan.why.append(f"{rel}: test file added or removed")
                    else:
                        plan.mode = "B"
                        plan.cands |= set(S.all)
                        plan.why.append(f"{rel}: module added or removed")
                    continue
                analyze_python(plan, rel, old, new, changed)
            else:
                if not rel.endswith(".sql"):
                    plan.reuse_ok = False
                if plan.mode == "none":
                    plan.mode = "files"
                plan.why.append(f"{rel}: data")
    except Exception as e:
        return force_all(plan, "analysis failed: " + repr(e)[:200])
    if plan.mode == "B" and any(module_name(r) in S.maps["core_modules"] for r in plan.changes if r.endswith(".py")):
        plan.cands.update(S.lazy_load)
    if plan.mode == "none" and plan.cands and plan.changes:
        plan.mode = "files"            # nothing executable changed, but some test reads the file itself
    if plan.mode != "none" and (plan.cands or plan.statediff):
        plan.cands.update(S.unknown)
    if plan.mode in ("A", "files") and not plan.cands:
        plan.mode = "none"
    if plan.mode == "B" and not plan.cands and not plan.statediff:
        plan.mode = "none"
    return plan


def force_all(plan, why):
    plan.mode = "all"
    plan.why.append(why)
    return plan


# ---------------------------------------------------------------- warm process

def ensure_warm():
    if S.warm:
        return
    C.warm()
    S.warm = True
    idx = {}
    root = C.ROOT + "/"
    for o in gc.get_objects():
        if type(o) is types.FunctionType:
            c = o.__code__
            if c.co_filename.startswith(root):
                idx.setdefault((c.co_filename[len(root):], c.co_firstlineno, c.co_name), []).append(o)
    S.fidx = idx
    S.runner = C.Runner()
    S.runner.after_setup = after_setup
    S.runner.instrument = S.stmt
    gc.freeze()


def after_setup(c):
    if (c.__module__, c.__name__) in C.MEMO_CONNS:
        if S.memo is None:
            try:
                with open(os.path.join(C.MAPS, "duck.pkl"), "rb") as f:
                    S.memo = pickle.load(f)
            except Exception as e:
                C.log("no duck memo", repr(e)[:100])
                S.memo = {}
        if S.memo and S.use_memo:
            C.wrap_executor(c, S.memo)


def import_tests(numbers):
    for n in numbers:
        try:
            S.runner.get_class(S.id_of[n])
        except BaseException as e:
            C.log("zygote cannot import", S.id_of[n], repr(e)[:100])


def class_key(n):
    return S.id_of[n].rsplit("::", 1)[0]


def setup_conflict(plan):
    """Does this change touch anything an expensive class set-up runs or reads?"""
    for k in S.expensive:
        info = S.maps["setup"][k]
        for rel, lines in plan.lines.items():
            f = S.maps["files"].get(rel)
            if f is not None and lines & info["lines"].get(f, set()):
                return True
        for rel in plan.changes:
            if rel in info["opened"] or (os.path.dirname(rel) + "/") in info["opened"]:
                return True
    return False


# ---------------------------------------------------------------- child side

def emit(fd, line):
    os.write(fd, (line + "\n").encode())


def rebase(ov):
    root = C.ROOT

    def fix(v):
        return ov + v[len(root):] if (v == root or v.startswith(root + "/")) else None

    for name, m in list(sys.modules.items()):
        if m is None or name.split(".")[0] not in ("sqlglot", "tests"):
            continue
        for k, v in list(vars(m).items()):
            if isinstance(v, str):
                nv = fix(v)
                if nv is not None:
                    setattr(m, k, nv)
            elif k == "__path__":
                try:
                    m.__path__ = [fix(p) or p for p in v]
                except Exception:
                    pass
            elif isinstance(v, type) and v.__module__ == name:
                for ak, av in list(vars(v).items()):
                    if isinstance(av, str):
                        nv = fix(av)
                        if nv is not None:
                            try:
                                setattr(v, ak, nv)
                            except Exception:
                                pass
    os.chdir(ov)
    os.environ["PYTHONPATH"] = ov
    sys.path[:] = [ov if p == root else p for p in sys.path]


_src_cache = {}


def _sources():
    if not _src_cache:
        for top in ("sqlglot", "tests"):
            for d, _, fs in os.walk(os.path.join(C.ROOT, top)):
                for f in fs:
                    if f.endswith(".py"):
                        p = os.path.join(d, f)
                        try:
                            with open(p, encoding="utf-8") as fh:
                                _src_cache[os.path.relpath(p, C.ROOT)] = fh.read()
                        except Exception:
                            pass
    return _src_cache


def ref_lines(name, only=None):
    """{rel: lines} of pinned source lines that mention `name` (in every file, or in `only`)."""
    pat = re.compile(r"(?<![A-Za-z0-9_])" + re.escape(name) + r"(?![A-Za-z0-9_])")
    out = {}
    for rel, text in _sources().items():
        if (only is not None and rel != only) or name not in text:
            continue
        ls = [i for i, line in enumerate(text.split("\n"), 1) if name in line and pat.search(line)]
        if ls:
            out[rel] = ls
    return out


def readers(name, only=None):
    """Tests that execute a line mentioning `name`, plus every test of a test file that mentions it outside a
    function (such a file reads the name while it is imported)."""
    refs = ref_lines(name, only)
    out = set()
    for rel, ls in refs.items():
        out |= tests_for_lines(rel, ls)
        if rel.startswith("tests/"):
            try:
                tree = ast.parse(_sources()[rel])
            except SyntaxError:
                continue
            inside = set()
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
                    inside.update(range(node.lineno, node.end_lineno + 1))
            if any(l not in inside for l in ls):
                out |= tests_importing(module_name(rel)) | set(S.by_file.get(rel, ()))
    return out, bool(refs)


def function_lines(key):
    """(rel, lines) of the function whose defaults are tracked under `key`, from the pinned import."""
    ent = S.fp0.get(statediff._INFO, {}).get(key) if S.fp0 else None
    if not ent:
        return None
    path, lines = ent
    if not path.startswith(C.ROOT + "/"):
        return None
    return path[len(C.ROOT) + 1:], lines


def suspects_from_state(changed_keys):
    """Tests that can see a changed class attribute or module global."""
    if len(changed_keys) > 400:
        return set(S.all), "many state changes"
    out = set()
    if len({k[2] for k in changed_keys}) > 60:
        return set(S.all), "many names changed"
    for kind, owner, name in changed_keys:
        dunder = name.startswith("__")
        if kind == "d":
            # a default argument changed: whoever runs that function can see it
            where = function_lines((kind, owner, name))
            if where is not None:
                out |= tests_for_lines(*where)
                continue
            kind = "c" if ":" in owner else "g"
        if kind == "g":
            # a module global is read by name inside its own module, and by attribute or import elsewhere
            rd, _ = readers(name) if not dunder else (set(), False)
            out |= rd
            continue
        rd, mentioned = readers(name) if not dunder else (set(), False)
        sid = S.maps["classes"].get(owner)
        users = members(sid) if sid is not None else None
        if users is None:
            out |= rd                           # never instantiated: enum, namespace
        elif mentioned:
            out |= users & rd
        else:
            out |= users                        # no textual reader: reached dynamically
    return out, f"{len(changed_keys)} state changes"


def include_cost(total, big):
    """Cost in ms of running tests of `total` ms, the longest taking `big`, on the best number of workers."""
    if total <= 0:
        return 0
    return min(C.WORKER_MS * k + max(big, -(-total // k)) for k in range(1, 9))


def run_policy(fd, cands, runner, calls=None, min_evidence=5):
    """Verifies suspects cheapest first, and stops verifying where including a test costs less than running it.
    `calls` maps a test to the pool calls and statement units that must really run; its other units are
    skipped or reuse recorded results."""
    ms = S.ms
    v = S.v
    if calls:
        v = dict(v)
        for n, (aff, units) in calls.items():
            frac = S.maps["call_frac"].get(S.num2idx[n])
            if frac:
                nc = S.maps["unit_base"][S.num2idx[n]][1]
                saved = sum(f for k, f in enumerate(frac[:nc]) if k not in aff)
                if units is not None:
                    saved += sum(f for k, f in enumerate(frac[nc:]) if k not in units)
                v[n] = S.v[n] * max(0.05, 1.0 - saved)
    order = sorted((n for n in cands if S.base_ok.get(n, True)), key=lambda n: v[n])
    remaining = set(cands)                       # not shown to pass
    rem_total = sum(ms[n] for n in remaining)
    rem_big = max((ms[n] for n in remaining), default=0)
    fails = []
    # Tests that are expensive to verify. When enough cheap tests run the same changed code and all pass, an
    # expensive one is taken to pass too (measured on the pinned commit: 44 of 44 synthetic changes).
    costly = set()
    if S.risk and min_evidence is not None:
        costly = {n for n in order if v[n] > COSTLY_S and S.id_of[n].endswith(ASSUME_PASS)}
    cheap_passed = 0
    cheap_skipped = False
    k = len(order)
    suf_v = [0.0] * (k + 1)                      # CPU still needed to verify order[i:], costly ones aside
    suf_all = [0.0] * (k + 1)
    for i in range(k - 1, -1, -1):
        n = order[i]
        suf_v[i] = suf_v[i + 1] + (0.0 if n in costly else v[n])
        suf_all[i] = suf_all[i + 1] + v[n]
    dirty = False
    for i, n in enumerate(order):
        if dirty or time.time() - T0 > S.deadline:
            break
        if not fails:
            now = include_cost(rem_total, rem_big) / 1000
            if n in costly:
                if cheap_passed >= min_evidence and not cheap_skipped:
                    remaining.discard(n)
                    rem_total -= ms[n]
                    if ms[n] >= rem_big:
                        rem_big = max((ms[x] for x in remaining), default=0)
                    emit(fd, f"P {n}")
                    C.log(f"   assumed to pass on {cheap_passed} cheap tests: {S.id_of[n]}")
                    continue
                if suf_all[i] > 1.1 * now + 0.05:
                    break
            elif suf_v[i] > 1.1 * now + 0.05:
                cheap_skipped = True
                break
        else:
            big2 = rem_big if ms[n] < rem_big else max((ms[x] for x in remaining if x != n), default=0)
            delta = (include_cost(rem_total, rem_big) - include_cost(rem_total - ms[n], big2)) / 1000
            if 0.5 * delta <= 0.6 * v[n]:
                continue
        limit = max(3.0, 3.0 * max(v[n], ms[n] / 1000) + 2.0)
        C.FC.reuse = C.FC.affected = None
        stmts.SU.affected = None
        if calls and n in calls:
            stmts.SU.affected = calls[n][1]
            if S.maps["unit_base"][S.num2idx[n]][1]:
                try:
                    with open(os.path.join(C.MAPS, "fut", f"{S.num2idx[n]}.pkl"), "rb") as f:
                        C.FC.reuse = pickle.load(f)
                    C.FC.affected = calls[n][0]
                except Exception:
                    C.FC.reuse = C.FC.affected = None
        t_run = time.process_time()
        try:
            C.arm(60.0)
            runner.prepare(S.id_of[n])
            C.arm(limit)
            ok, err = runner.run(S.id_of[n])
            C.disarm()
            if S.debug and time.process_time() - t_run > 0.25:
                C.log(f"   {time.process_time() - t_run:5.2f}s (expected {v[n]:.2f}) {S.id_of[n]}")
        except C.Timeout:
            ok, err = False, "timeout"
            dirty = True
        except BaseException as e:
            C.disarm()
            ok, err = False, "crash " + repr(e)[:100]
        if ok:
            remaining.discard(n)
            rem_total -= ms[n]
            if ms[n] >= rem_big:
                rem_big = max((ms[x] for x in remaining), default=0)
            emit(fd, f"P {n}")
            if n not in costly:
                cheap_passed += 1
        else:
            fails.append(n)
            emit(fd, f"F {n}")
            C.log(f"   fail {S.id_of[n]}: {(err or '').strip().splitlines()[-1][:160] if err else ''}")


def _no_statediff(plan):
    plan.statediff = False
    plan.reuse_ok = False
    plan.broken = False
    return plan


def child_main(fd, plan):
    cands = set(plan.cands)
    state_keys = None
    ov = None
    if plan.changes:
        ov = os.path.join(C.TMP, f"ov_{os.getpid()}")
        C.build_overlay(C.ROOT, ov, plan.changes)
    # DuckDB answers are keyed by query text, so they stay valid unless the data files themselves change.
    S.use_memo = not any(r.startswith("tests/fixtures/") and not r.endswith(".sql") for r in plan.changes)
    if plan.mode == "B":
        try:
            failed = C.reimport(ov)
        except BaseException as e:
            C.log("   sqlglot does not import:", repr(e)[:200])
            emit(fd, "C " + " ".join(map(str, S.all)))
            return
        if failed:
            # A module that no longer imports breaks every test that would load it, including tests that
            # only look a dialect up by name. The maps cannot tell those apart, so every test is a suspect.
            cands = set(S.all)
            plan = _no_statediff(plan)
        if plan.statediff:
            try:
                keys = statediff.diff(S.fp0, statediff.snapshot(S.maps["unstable"]), S.maps["lazy"])
                extra, why = suspects_from_state(keys)
                if len(keys) <= 60:
                    state_keys = keys
                C.log(f"   state: {why}: {sorted(keys)[:6]} -> {len(extra)} tests")
                cands |= extra
            except BaseException as e:
                C.log("   state diff failed:", repr(e)[:200])
                cands = set(S.all)
            if cands:
                cands.update(S.unknown)
        if plan.broken and len(cands) < len(S.all):
            # every test of a file that can no longer be imported fails
            for f, numbers in S.by_file.items():
                try:
                    importlib.import_module(f[:-3].replace("/", "."))
                except BaseException:
                    cands.update(numbers)
        runner = C.Runner()
        runner.after_setup = after_setup
        runner.instrument = S.stmt and plan.reuse_ok
    else:
        for rel, pairs in plan.swaps:
            swapped = 0
            for o, n in pairs:
                for f in S.fidx.get((rel, o.co_firstlineno, o.co_name), ()):
                    if f.__code__.co_freevars != n.co_freevars:
                        raise NeedReimport("free variables changed")
                    f.__code__ = n
                    swapped += 1
            if not swapped and cands:
                raise NeedReimport("no live function to patch in " + rel)
        runner = S.runner
        if ov is not None:
            rebase(ov)
        if plan.mode == "files" or S.conflict_now:
            # class set-ups done in the warm process may not hold for this change: redo them here
            runner = C.Runner()
            runner.classes = dict(S.runner.classes)
            runner.after_setup = after_setup
            runner.instrument = S.runner.instrument
    emit(fd, "C " + " ".join(map(str, sorted(cands))))
    # Recorded pool-call results may stand in for calls that run no changed statement. A changed .sql fixture
    # reaches a call only through its pickled arguments, which are compared; any other changed file rules it out.
    calls = None
    only_sql = all(r.endswith(".sql") for r in plan.changes)
    if only_sql and (plan.mode == "files" or (plan.mode == "A" and not S.conflict_now)):
        calls = affected_calls(plan, cands, 0, S.stmt and plan.mode == "A" and ov is None)
    elif plan.mode == "B" and plan.reuse_ok and state_keys is not None:
        try:
            calls = affected_calls(plan, cands, state_mask(state_keys), S.stmt)
        except Exception as e:
            C.log("   no reuse:", repr(e)[:100])
            calls = None
    run_policy(fd, cands, runner, calls, min_evidence=5 if plan.mode == "A" else None)


def run_child(plan):
    """Returns (cands, passed, failed) or None when the child gave no usable answer."""
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        code = 0
        try:
            os.close(r)
            child_main(w, plan)
            emit(w, "D")
        except NeedReimport as e:
            C.log("   hot swap not possible:", e)
            emit(w, "X")
            code = 2
        except BaseException as e:
            C.log("   child error:", repr(e)[:300])
            code = 1
        finally:
            try:
                shutil.rmtree(os.path.join(C.TMP, f"ov_{os.getpid()}"), ignore_errors=True)
            except Exception:
                pass
            os._exit(code)
    os.close(w)
    buf = b""
    hard = time.time() + max(5.0, min(90.0, S.hard - (time.time() - T0)))
    while True:
        left = hard - time.time()
        if left <= 0:
            try:
                os.kill(pid, 9)
            except OSError:
                pass
            break
        ready, _, _ = select.select([r], [], [], min(left, 5))
        if ready:
            chunk = os.read(r, 65536)
            if not chunk:
                break
            buf += chunk
    os.close(r)
    try:
        os.waitpid(pid, 0)
    except OSError:
        pass
    shutil.rmtree(os.path.join(C.TMP, f"ov_{pid}"), ignore_errors=True)
    cands = None
    passed, failed = set(), set()
    for line in buf.decode().split("\n"):
        if line == "X":
            return "reimport"
        if line.startswith("C"):
            cands = {int(x) for x in line[1:].split()}
        elif line.startswith("P "):
            passed.add(int(line[2:]))
        elif line.startswith("F "):
            failed.add(int(line[2:]))
    if cands is None:
        return None
    return cands, passed, failed


# ---------------------------------------------------------------- driver

def decide(plan):
    """Returns the selected test numbers for one commit."""
    if plan.mode == "none":
        return set(plan.forced)
    if plan.mode == "all":
        return set(S.all)
    if time.time() - T0 > S.deadline:
        return set(S.all) if (plan.mode == "B" and plan.statediff) else plan.cands | plan.forced
    ensure_warm()
    S.conflict_now = setup_conflict(plan)
    if plan.mode == "B":
        if plan.statediff and S.fp0 is None:
            S.fp0 = statediff.snapshot(S.maps["unstable"])
    else:
        import_tests(sorted(plan.cands))
        if not S.conflict_now:
            for n in sorted(plan.cands):
                k = class_key(n)
                if k in S.expensive and k not in S.prepared:
                    S.use_memo = True
                    try:
                        c, _ = S.runner.get_class(S.id_of[n])
                        S.runner.ensure_setup(c)
                    except BaseException as e:
                        C.log("set-up in warm process failed", k, repr(e)[:100])
                    S.prepared.add(k)
    res = run_child(plan)
    if res == "reimport":
        plan.mode = "B"
        plan.statediff = True
        plan.reuse_ok = False
        if S.fp0 is None:
            S.fp0 = statediff.snapshot(S.maps["unstable"])
        res = run_child(plan)
        if res == "reimport":
            res = None
    if res is None:
        C.log("   no answer from child; selecting conservatively")
        return set(S.all) if plan.mode == "B" else plan.cands | plan.forced
    cands, passed, failed = res
    return (cands - passed) | failed | plan.forced


def fallback(out_fd):
    """Last resort: every test on every commit. Never a miss, never a rejected run."""
    tests = json.load(open(os.path.join(C.DATA, "tests.json")))["tests"]
    ms = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
    _, workers = C.pack(sorted(ms), ms)
    cdir = os.path.join(C.DATA, "commits")
    commits = sorted(f[:-6] for f in os.listdir(cdir) if f.endswith(".patch"))
    lines = [json.dumps({"commit": c, "workers": workers}, separators=(",", ":")) for c in commits]
    os.write(out_fd, ("\n".join(lines) + "\n").encode() if lines else b"")


def main():
    out_fd = os.dup(1)
    os.dup2(2, 1)
    try:
        run(out_fd)
    except BaseException as e:
        C.log("selector failed, selecting everything:", repr(e)[:300])
        fallback(out_fd)


def run(out_fd):
    load()
    cdir = os.path.join(C.DATA, "commits")
    commits = sorted(f[:-6] for f in os.listdir(cdir) if f.endswith(".patch"))
    # Wall-clock guards. The batch limit is 180 s for about 40 commits and 270 s for about 60.
    S.hard = float(os.environ.get("BR_HARD", min(250.0, 4.1 * max(len(commits), 10))))
    S.deadline = 0.6 * S.hard
    S.use_memo = True
    S.conflict_now = False
    S.risk = os.environ.get("BR_RISK", "1") != "0"
    S.stmt = os.environ.get("BR_STMT", "1") != "0"
    S.debug = os.environ.get("BR_DEBUG", "0") != "0"
    S.no_stmt = set(S.maps.get("no_stmt", ()))
    plans = []
    for c in commits:
        with open(os.path.join(cdir, c + ".patch"), "rb") as f:
            text = f.read().decode("utf-8", errors="surrogateescape")
        plans.append(analyze(c, text))
    t_an = time.time() - T0
    # Changes that touch an expensive class set-up go first, before the warm process runs that set-up itself.
    order = sorted(range(len(plans)), key=lambda i: (0 if (plans[i].mode in ("A", "files") and plans[i].lines
                                                           and _conflicts(plans[i])) else 1, i))
    answers = {}
    for i in order:
        p = plans[i]
        t1 = time.time()
        try:
            sel = decide(p)
        except BaseException as e:
            C.log("decide failed", p.commit, repr(e)[:200])
            sel = set(S.all)
        sel = {n for n in sel if n in S.ms}
        cost, workers = C.pack(sorted(sel), S.ms)
        answers[p.commit] = workers
        C.log(f"{p.commit} {p.mode:5s} cands={len(p.cands):4d} selected={len(sel):4d} cost={cost / 1000:6.2f} "
              f"t={time.time() - t1:5.2f}s  {'; '.join(p.why)[:110]}")
    lines = [json.dumps({"commit": c, "workers": answers[c]}, separators=(",", ":")) for c in commits]
    os.write(out_fd, ("\n".join(lines) + "\n").encode())
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    C.log(f"analysis {t_an:.2f}s  wall {time.time() - T0:.1f}s  cpu {ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime:.2f}s")


def _conflicts(plan):
    try:
        return setup_conflict(plan)
    except Exception:
        return True


if __name__ == "__main__":
    main()
