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
from br import dual
from br import keytrack
from br import statediff
from br import stmts

T0 = time.time()
PYTEST_CONFIG = re.compile(r"^\[(tool\.pytest|pytest|tool:pytest)", re.M)
CONFIG_FILES = {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"}
HEAVY_MS = 3000
COSTLY_S = 3.0
# (runs of the changed comparison, passing tests) needed before stopping early; "" turns it off
DUAL_STOP = "30,3"
# The only test the skip rule was measured for.
ASSUME_PASS = ("::TestExecutor::test_execute_tpcds",)


class State:
    pass


class NeedReimport(Exception):
    pass


PHASES = {}


def phase(name, t0):
    """Adds CPU spent since t0 to a named bucket (reported with BR_DEBUG)."""
    PHASES[name] = PHASES.get(name, 0.0) + time.process_time() - t0


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
    S.key_read = frozenset(m.get("key_read", ()))
    S.key_table = (m.get("key_table") or {}) if "key_read" in m else {}
    S.key_untracked = frozenset(S.idx2num[i] for i in m.get("key_untracked", ()) if S.idx2num[i] is not None) \
        | frozenset(S.unknown)
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


def line_mask(lines_by_file, which="ulines"):
    mask = 0
    for rel, lines in lines_by_file.items():
        ul = S.maps[which].get(S.maps["files"].get(rel), {})
        for l in lines:
            j = ul.get(l)
            if j is not None:
                mask |= _bits(j)
    return mask


def state_mask(changed_keys):
    """Pool calls and statement units that can see a changed class attribute or module global, as
    (all of them, those whose later stage can). A unit sees a class attribute when it uses the class and runs
    a line that names the attribute; when no line names it, using the class is enough."""
    mask = 0
    late = 0
    for kind, owner, name in changed_keys:
        if kind == "d":
            where = function_lines((kind, owner, name))
            if where is not None:
                lines = {where[0]: where[1]}
                mask |= line_mask(lines)
                late |= line_mask(lines, "ulines2")
                continue
            kind = "c" if ":" in owner else "g"
        refs = ref_lines(name) if not name.startswith("__") else {}
        if kind == "c" and S.maps["classes"].get(owner) is not None:
            j = S.maps["uclasses"].get(owner)
            users = _bits(j) if j is not None else 0
            if refs:
                mask |= users & line_mask(refs)
                late |= users & line_mask(refs, "ulines2")
            else:
                mask |= users
                late |= users
        else:
            mask |= line_mask(refs)
            late |= line_mask(refs, "ulines2")
    return mask, late


def gen_replay_ok(plan):
    """Can recorded code generator calls be replayed for this change? Not when it changes a line the replay
    itself runs besides those calls (unpickling the inputs, making the generators)."""
    check = S.maps.get("gen_check")
    if check is None or "ulines3" not in S.maps:
        return False
    for rel, lines in plan.lines.items():
        f = S.maps["files"].get(rel)
        if f is not None and check.get(f, set()) & set(lines):
            return False
    return True


def affected_calls(plan, cands=None, extra_mask=0, with_stmts=False, probes=False, extra_late=None,
                   gen_probes=False):
    """{test number: (pool-call indexes that must really run, statement units that must run, calls that may
    stop after their first stage, those of them that also replay their code generator calls)}, for tests that
    make pool calls."""
    mask = line_mask(plan.lines) | extra_mask
    # calls whose later stage (after the optimizer run inside execute) can see the change cannot stop early
    late = (line_mask(plan.lines, "ulines2") | (extra_mask if extra_late is None else extra_late)) if probes else None
    # ... unless it is seen only inside the executor's code generator calls, which can be replayed
    late3 = line_mask(plan.lines, "ulines3") if (probes and gen_probes and extra_mask == 0 and extra_late is None
                                                  and gen_replay_ok(plan)) else None
    out = {}
    for n in (plan.cands if cands is None else cands):
        i = S.num2idx.get(n)
        ub = S.maps["unit_base"].get(i)
        if ub is None:
            continue
        base, nc, ns = ub
        part = (mask >> base) & ((1 << (nc + ns + 1)) - 1)
        calls = {k for k in range(nc) if (part >> k) & 1}
        # When the change is also seen outside the units (set-up, assignments between the statements), objects
        # made there may carry it into any statement, so no statement is skipped. Pool calls are safe either
        # way: everything they are given is pickled and compared.
        rest_sees_it = (part >> (nc + ns)) & 1
        units = None
        if with_stmts and ns and i not in S.no_stmt and not rest_sees_it:
            units = {k for k in range(ns) if (part >> (nc + k)) & 1}
        probe = set()
        probe2 = set()
        if late is not None:
            part2 = (late >> base) & ((1 << nc) - 1)
            probe = {k for k in calls if not (part2 >> k) & 1}
            if late3 is not None:
                part3 = (late3 >> base) & ((1 << nc) - 1)
                probe2 = {k for k in calls if k not in probe and not (part3 >> k) & 1}
        out[n] = (calls, units, probe | probe2, probe2)
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
        self.swapped = set()      # A: files whose only change is function code that is patched in memory
        self.dual = False         # A: the one changed comparison is counted (see dual.py)
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
    if not structure and S.nesting and not in_tests:
        # Code objects cannot be paired one to one. When the difference is confined to function bodies (a
        # lambda or generator expression inside a function was added or removed), each such function is
        # still paired with its new version by position in its class or module, and is replaced as a whole.
        pairs = []
        try:
            body_changed = C.diff_functions(ocode, ncode, pairs)
        except ValueError:
            body_changed = True
        if not body_changed and pairs and not any((rel, o.co_firstlineno, o.co_name) in imp for o, _ in pairs) \
                and not any(x.co_flags & 1 and (rel, x.co_firstlineno, x.co_name) in imp
                            for o, _ in pairs for x in C.all_code(o)):
            lines = set()
            for o, _ in pairs:
                lines |= C.all_lines(o)
            plan.cands |= tests_for_lines(rel, lines)
            plan.lines.setdefault(rel, set()).update(lines)
            plan.swaps.append((rel, pairs))
            if not openers(rel):
                plan.swapped.add(rel)
            if plan.mode in ("none", "files"):
                plan.mode = "A"
            plan.why.append(f"{rel}: {len(pairs)} function(s), nesting changed inside")
            return True
    if not structure:
        # A function, lambda or class was added or removed, so old and new code objects cannot be paired by
        # position. A test can be affected through a pinned function that is gone or different (found by the
        # lines it runs), or through what the import computes (found by comparing import state).
        if S.nesting:
            # Units of a test that run none of these functions and cannot see a changed class attribute or
            # global behave as on the pinned commit, as for any other class-level change.
            lines = C.dropped_function_lines(ocode, ncode)
            plan.lines.setdefault(rel, set()).update(lines)
        else:
            plan.reuse_ok = False
            lines = set(changed)
            spans = statement_spans(old)
            for l in changed:
                lines |= enclosing_lines(ocode, l) | statement_lines(spans, l)
        plan.cands |= tests_for_lines(rel, lines)
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
        if S.dual and len(own) == 1:
            counted = dual.instrument(old, new, path)
            if counted is not None:
                out2 = []
                try:
                    C.diff_tree(ocode, counted, [], out2)
                    pairs2 = [(o, n) for o, n, _, p in out2 if p and C.is_function(o)]
                    if len(pairs2) == len(pairs) and all(a[0] is b[0] for a, b in zip(pairs, pairs2)):
                        pairs = pairs2
                        plan.dual = True
                except ValueError:
                    pass
        plan.swaps.append((rel, pairs))
        if not openers(rel):
            plan.swapped.add(rel)     # nobody reads this file's text, so no tree on disk has to show the change
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
    t0 = time.process_time()
    C.warm()
    phase("warm import", t0)
    t0 = time.process_time()
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
    phase("function index", t0)


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


def _table_readers(table, changed, resized, nested):
    """Build indexes of the tests that asked table `table` for a key in `changed`, took its size (when
    `resized`), or looked at it as a whole. `nested` gives, for some changed keys, the readers of the inner
    table behind that key; those replace the readers of the key itself."""
    if table not in S.key_read:
        return set()                             # the build saw no test read this table at all
    with open(os.path.join(C.MAPS, "keys", f"{table}.pkl"), "rb") as f:      # a missing file is an error
        whole, size, keys = pickle.load(f)
    idx = set(array("H", whole))
    if resized:
        idx.update(array("H", size))
    for h in changed:
        inner = nested.get(h)
        if inner is not None:
            idx |= inner
        else:
            b = keys.get(h)
            if b is not None:
                idx.update(array("H", b))
    return idx


def entry_readers(key, old_modules):
    """Tests that can see how the dict table behind state key `key` changed: those that asked it for an entry
    that differs, took its size when the size differs, or looked at it as a whole. None when the build has no
    read record for that table or the two values cannot be compared entry by entry."""
    table = S.key_table.get(key)
    if table is None or old_modules is None:
        return None
    try:
        old = keytrack.lookup(old_modules, key)
        new = keytrack.lookup(sys.modules, key)
        core = old_modules.get("sqlglot.expressions.core")
        old_expr = getattr(core, "Expr", None)
        unordered = key in S.maps["unstable"]
        ch = keytrack.changed_entries(old, new, old_expr, unordered)
        if ch is None:
            return None
        changed, resized = ch
        nested = {}
        for h, (ov, nv) in changed.items():
            inner = S.key_table.get(key + (h,))
            if inner is not None and ov is not keytrack.MISSING and nv is not keytrack.MISSING:
                ch2 = keytrack.changed_entries(ov, nv, old_expr, unordered)
                if ch2 is not None:
                    nested[h] = _table_readers(inner, ch2[0], ch2[1], {})
        idx = _table_readers(table, changed, resized, nested)
    except Exception as e:
        C.log("   entry comparison failed:", key, repr(e)[:120])
        return None
    return {S.idx2num[i] for i in idx if S.idx2num[i] is not None}


def suspects_from_state(changed_keys, old_modules=None):
    """Tests that can see a changed class attribute or module global."""
    if len(changed_keys) > 400:
        return set(S.all), "many state changes"
    out = set()
    narrowed = 0
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
            sus, _ = readers(name) if not dunder else (set(), False)
            if S.keys_on:
                seen = entry_readers((kind, owner, name), old_modules)
                if seen is not None:
                    narrowed += len(sus) - len(seen | (sus & S.key_untracked))
                    sus = seen | (sus & S.key_untracked)
            out |= sus
            continue
        rd, mentioned = readers(name) if not dunder else (set(), False)
        sid = S.maps["classes"].get(owner)
        users = members(sid) if sid is not None else None
        if users is None:
            sus = rd                            # never instantiated: enum, namespace
        elif mentioned:
            sus = users & rd
        else:
            sus = users                         # no textual reader: reached dynamically
        if S.keys_on:
            seen = entry_readers(("c", owner, name), old_modules)
            if seen is not None:
                # tests whose reads were not recorded stay suspects by the rule above
                narrowed += len(sus) - len(seen | (sus & S.key_untracked))
                sus = seen | (sus & S.key_untracked)
        out |= sus
    return out, f"{len(changed_keys)} state changes" + (f", entry reads spare {narrowed} tests" if narrowed else "")


def _imported_names(node, modname, is_package):
    """Absolute names of the modules an import statement can load."""
    if isinstance(node, ast.Import):
        out = []
        for a in node.names:
            parts = a.name.split(".")
            out.extend(".".join(parts[:i]) for i in range(1, len(parts) + 1))
        return out
    base = node.module or ""
    if node.level:
        pkg = modname if is_package else modname.rpartition(".")[0]
        for _ in range(node.level - 1):
            pkg = pkg.rpartition(".")[0]
        base = f"{pkg}.{base}" if base else pkg
    return [base] + [f"{base}.{a.name}" for a in node.names]


def users_of_failed(failed, old_modules):
    """Tests that can notice that the modules in `failed` no longer import, or None when that cannot be
    narrowed down. A test needs such a module when it runs code of it, makes an object of one of its classes,
    runs an import statement or a line that names one of its classes, or reads an entry that is now missing
    from a registry (the dialect registry is a table whose reads were recorded). Test files that import one
    are probed by the caller."""
    bad = set(failed)
    if S.fp0 is None or bad & set(S.maps["core_modules"]):
        return None                              # plain `import sqlglot` is broken: nearly everything fails
    out = set()
    for m in bad:
        mod = old_modules.get(m)
        path = getattr(mod, "__file__", None)
        if mod is None or not path or not path.startswith(C.ROOT + "/"):
            return None
        f = S.maps["files"].get(path[len(C.ROOT) + 1:])
        if f is not None:
            for sid in set(S.maps["lines"].get(f, {}).values()):
                out |= members(sid)
        for name, v in list(vars(mod).items()):
            if isinstance(v, type) and v.__module__ == m:
                out |= readers(name)[0]
    for key, sid in S.maps["classes"].items():
        if key.split(":", 1)[0] in bad:
            out |= members(sid)
    # Tests that asked for a failed module by name (import statement, __import__, importlib.import_module), or
    # listed the directory it lives in (pkgutil and the like find modules that way).
    dirs = set()
    for m in bad:
        path = old_modules[m].__file__[len(C.ROOT) + 1:]
        d = os.path.dirname(path)
        dirs.add(d + "/")
        if path.endswith("__init__.py"):
            dirs.add(os.path.dirname(d) + "/")
    for key, sid in S.maps["opened"].items():
        if key.startswith("import:"):
            name = key[7:]
            if any(name == m or name.startswith(m + ".") for m in bad):
                out |= members(sid)
        elif key in dirs:
            out |= members(sid)
    for rel, text in _sources().items():
        if "import" not in text:
            continue
        try:
            tree = ast.parse(text)
        except SyntaxError:
            continue
        modname = module_name(rel)
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)) and \
                    bad.intersection(_imported_names(node, modname, rel.endswith("__init__.py"))):
                out |= tests_for_lines(rel, range(node.lineno, node.end_lineno + 1))
    # What the modules that still import hold now: registries without the failed modules' classes, and
    # whatever else the patch changed.
    keys = [k for k in statediff.diff(S.fp0, statediff.snapshot(S.maps["unstable"]), S.maps["lazy"])
            if k[1].split(":", 1)[0] not in bad]
    extra, why = suspects_from_state(keys, old_modules)
    C.log(f"   state outside the failed modules: {why}: {sorted(keys)[:4]} -> {len(extra)} tests")
    return out | extra


GC_LIMIT = 1500 << 20      # bytes of resident memory above which garbage is collected (the run has 4 GB)
_PAGE = os.sysconf("SC_PAGE_SIZE")


def mem_guard():
    """The cyclic garbage collector is off (it costs about a fifth of the CPU of a test like TPC-DS, and every
    commit runs in a short-lived fork); garbage is collected only when resident memory grows large."""
    if not S.gc_off:
        return
    try:
        with open("/proc/self/statm", "rb") as f:
            rss = int(f.read().split()[1]) * _PAGE
    except Exception:
        rss = GC_LIMIT + 1
    if rss > GC_LIMIT:
        gc.collect()


def include_cost(total, big):
    """Cost in ms of running tests of `total` ms, the longest taking `big`, on the best number of workers."""
    if total <= 0:
        return 0
    return min(C.WORKER_MS * k + max(big, -(-total // k)) for k in range(1, 9))


def run_policy(fd, cands, runner, calls=None, min_evidence=5, dual_stop=None):
    """Verifies suspects cheapest first, and stops verifying where including a test costs less than running it.
    `calls` maps a test to the pool calls and statement units that must really run; its other units are
    skipped or reuse recorded results."""
    ms = S.ms
    v = S.v
    if calls:
        v = dict(v)
        for n, (aff, units, probe, _) in calls.items():
            frac = S.maps["call_frac"].get(S.num2idx[n])
            if frac:
                nc = S.maps["unit_base"][S.num2idx[n]][1]
                saved = sum(f for k, f in enumerate(frac[:nc]) if k not in aff)
                saved += 0.8 * sum(frac[k] for k in probe)        # the first stage is about a fifth of a call
                if units is not None:
                    saved += sum(f for k, f in enumerate(frac[nc:]) if k not in units)
                v[n] = S.v[n] * max(0.05, 1.0 - saved)
    testable = [n for n in cands if S.base_ok.get(n, True)]
    remaining = set(cands)                       # not shown to pass
    rem_total = sum(ms[n] for n in remaining)
    rem_big = max((ms[n] for n in remaining), default=0)
    fails = []
    # Tests that are expensive to verify. When enough cheap tests run the same changed code and all pass, an
    # expensive one is taken to pass too (measured on the pinned commit: 44 of 44 synthetic changes).
    costly = set()
    if S.risk and min_evidence is not None:
        costly = {n for n in testable if v[n] > COSTLY_S and S.id_of[n].endswith(ASSUME_PASS)}
    # costly tests go last: the rule above needs every cheap test to have passed first
    order = sorted(testable, key=lambda n: (n in costly, v[n]))
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
        if dirty or S.no_verify or time.time() - T0 > S.deadline:
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
            # Worth verifying when the expected saving beats the expected CPU: a test that passes costs its
            # full time to verify, one that fails stops at its first failing part (taken as 2 s at most).
            if S.debug and ms[n] >= HEAVY_MS:
                C.log(f"   after a failure: {S.id_of[n].rsplit('::', 1)[1]} saves {delta:.1f}s if it passes, "
                      f"costs {v[n]:.1f}s to verify (unreduced {S.v[n]:.1f}s)")
            if delta <= v[n] + min(v[n], 2.0):
                continue
        limit = max(3.0, 3.0 * max(v[n], ms[n] / 1000) + 2.0)
        C.FC.reuse = C.FC.affected = C.FC.probe = C.FC.probe2 = None
        stmts.SU.affected = None
        stmts.SU.known = None
        if calls and n in calls:
            stmts.SU.affected = calls[n][1]
            stmts.SU.known = S.maps["unit_base"][S.num2idx[n]][2]
            if S.maps["unit_base"][S.num2idx[n]][1]:
                try:
                    with open(os.path.join(C.MAPS, "fut", f"{S.num2idx[n]}.pkl"), "rb") as f:
                        C.FC.reuse = pickle.load(f)
                    C.FC.affected = calls[n][0]
                    C.FC.probe = calls[n][2]
                    C.FC.probe2 = calls[n][3]
                except Exception:
                    C.FC.reuse = C.FC.affected = C.FC.probe = C.FC.probe2 = None
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
        mem_guard()
        if ok:
            remaining.discard(n)
            rem_total -= ms[n]
            if ms[n] >= rem_big:
                rem_big = max((ms[x] for x in remaining), default=0)
            emit(fd, f"P {n}")
            if n not in costly:
                cheap_passed += 1
            # The patch swaps `is` and `==` (or their negations), every run so far compared against None,
            # True, False, a class or an enum member, and old and new operator never disagreed: every test so
            # far behaved exactly as on the pinned commit, and the rest are taken to do the same.
            if dual_stop and not fails and dual.Tally.diverged == 0 and dual.Tally.open_ended == 0 \
                    and dual.Tally.count >= dual_stop[0] and cheap_passed >= dual_stop[1]:
                rest = [m for m in order[i + 1:] if m in remaining]
                for m in rest:
                    emit(fd, f"P {m}")
                C.log(f"   old and new operator agreed {dual.Tally.count} times in {cheap_passed} tests; "
                      f"{len(rest)} remaining tests taken to pass")
                break
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
    tc = time.process_time()
    # Files a test may read from disk. In mode A the patched functions live in memory, so only other changed
    # files (fixtures, sources some test opens) need a tree that shows them.
    on_disk = dict(plan.changes) if plan.mode != "A" else \
        {r: d for r, d in plan.changes.items() if r not in plan.swapped}
    if on_disk:
        ov = os.path.join(C.TMP, f"ov_{os.getpid()}")
        C.build_overlay(C.ROOT, ov, plan.changes)
    # DuckDB answers are keyed by query text, so they stay valid unless the data files themselves change.
    S.use_memo = not any(r.startswith("tests/fixtures/") and not r.endswith(".sql") for r in plan.changes)
    if plan.mode == "B":
        old_modules = {k: m for k, m in sys.modules.items() if k == "sqlglot" or k.startswith("sqlglot.")}
        try:
            failed = C.reimport(ov)
        except BaseException as e:
            C.log("   sqlglot does not import:", repr(e)[:200])
            emit(fd, "C " + " ".join(map(str, S.all)))
            return
        emit(fd, f"T re-import {time.process_time() - tc:.3f}")
        tc = time.process_time()
        if failed:
            # modules inside a package that no longer imports cannot import either
            failed = sorted(set(failed) | {k for k in old_modules if any(k.startswith(f + ".") for f in failed)})
            # A module that no longer imports breaks every test that would load it, including tests that
            # only look a dialect up by name.
            users = None
            if S.failed_on:
                try:
                    users = users_of_failed(failed, old_modules)
                except BaseException as e:
                    C.log("   failed modules not narrowed:", repr(e)[:200])
            plan = _no_statediff(plan)
            if users is None or len(users) > S.failed_cap * len(S.all):
                cands = set(S.all)
            else:
                cands |= users
                cands.update(S.unknown)
                plan.broken = True               # test files that no longer import are found below
                C.log(f"   {len(failed)} modules do not import: {len(cands)} tests can notice")
        if plan.statediff:
            try:
                keys = statediff.diff(S.fp0, statediff.snapshot(S.maps["unstable"]), S.maps["lazy"])
                extra, why = suspects_from_state(keys, old_modules)
                if len(keys) <= 60:
                    state_keys = keys
                C.log(f"   state: {why}: {sorted(keys)[:6]} -> {len(extra)} tests")
                cands |= extra
            except BaseException as e:
                C.log("   state diff failed:", repr(e)[:200])
                cands = set(S.all)
            if cands:
                cands.update(S.unknown)
        if plan.statediff:
            emit(fd, f"T state diff {time.process_time() - tc:.3f}")
        tc = time.process_time()
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
                inner_o, inner_n = C.all_code(o)[1:], C.all_code(n)[1:]
                if [c.co_name for c in inner_o] != [c.co_name for c in inner_n]:
                    # the function is replaced as a whole; an inner function made from the old code earlier
                    # (a closure kept somewhere) would keep running the old code
                    for c in inner_o:
                        if c.co_flags & 1 and S.fidx.get((rel, c.co_firstlineno, c.co_name)):
                            raise NeedReimport("an inner function of the old code is alive")
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
    emit(fd, f"T child prepare {time.process_time() - tc:.3f}")
    tc = time.process_time()
    # Recorded pool-call results may stand in for calls that run no changed statement. A changed .sql fixture
    # reaches a call only through its pickled arguments, which are compared; any other changed file rules it out.
    calls = None
    only_sql = all(r.endswith(".sql") for r in on_disk)
    if only_sql and (plan.mode == "files" or (plan.mode == "A" and not S.conflict_now)):
        calls = affected_calls(plan, cands, 0, S.stmt and plan.mode == "A" and ov is None,
                               probes=plan.mode == "A" and ov is None,
                               gen_probes=plan.mode == "A" and ov is None and S.gen_replay)
    elif plan.mode == "B" and plan.reuse_ok and state_keys is not None:
        try:
            seen, seen_late = state_mask(state_keys)
            calls = affected_calls(plan, cands, seen, S.stmt, probes=True, extra_late=seen_late)
        except Exception as e:
            C.log("   no reuse:", repr(e)[:100])
            calls = None
    emit(fd, f"T unit selection {time.process_time() - tc:.3f}")
    tc = time.process_time()
    dual.Tally.count = dual.Tally.diverged = dual.Tally.open_ended = 0
    run_policy(fd, cands, runner, calls, min_evidence=5 if plan.mode == "A" else None,
               dual_stop=S.dual_stop if (plan.mode == "A" and plan.dual and S.risk) else None)
    emit(fd, f"T tests {plan.mode} {time.process_time() - tc:.3f}")


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
        if line.startswith("T "):
            name, _, sec = line[2:].rpartition(" ")
            PHASES[name] = PHASES.get(name, 0.0) + float(sec)
            continue
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
            t0 = time.process_time()
            S.fp0 = statediff.snapshot(S.maps["unstable"])
            phase("baseline state snapshot", t0)
    else:
        t0 = time.process_time()
        import_tests(sorted(plan.cands))
        phase("test module imports", t0)
        t0 = time.process_time()
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
        phase("class set-up in warm process", t0)
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
    S.dual = os.environ.get("BR_DUAL", "1") != "0"
    S.keys_on = os.environ.get("BR_KEYS", "1") != "0"
    S.nesting = os.environ.get("BR_NESTING", "1") != "0"
    S.failed_on = os.environ.get("BR_FAILED", "1") != "0"
    S.gen_replay = os.environ.get("BR_GEN", "1") != "0"
    S.no_verify = os.environ.get("BR_NOVERIFY", "0") != "0"
    S.gc_off = os.environ.get("BR_GC", "1") == "1"      # `BR_GC=0` keeps the garbage collector on
    if S.gc_off:
        gc.disable()
    S.failed_cap = float(os.environ.get("BR_FAILED_CAP", "1.0"))
    stop = os.environ.get("BR_DUAL_STOP", DUAL_STOP)
    S.dual_stop = tuple(int(x) for x in stop.split(",")) if stop else None
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
    if S.debug:
        C.log("phases: " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(PHASES.items(), key=lambda kv: -kv[1])))
    C.log(f"analysis {t_an:.2f}s  wall {time.time() - T0:.1f}s  cpu {ru.ru_utime + ru.ru_stime + rc.ru_utime + rc.ru_stime:.2f}s")


def _conflicts(plan):
    try:
        return setup_conflict(plan)
    except Exception:
        return True


if __name__ == "__main__":
    main()
