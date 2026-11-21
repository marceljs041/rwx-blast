"""Fingerprints of the imported sqlglot classes and module globals, to see what a re-import changed.

Function code is deliberately left out: a change inside a function body is found through line coverage. What
this catches is everything computed at import: class constants, tables built by metaclasses, registries."""
import enum
import functools
import logging
import re
import sys
import types

_PRIM = (bool, int, float, str, bytes, complex, type(None), type(Ellipsis))
_SKIP_CLASS = {"__dict__", "__weakref__", "__doc__", "__module__", "__qualname__", "__firstlineno__",
               "__static_attributes__", "__annotations__", "__parameters__", "__orig_bases__", "_abc_impl",
               "__annotate__", "__type_params__"}
_ADDR = re.compile(r"0x[0-9a-fA-F]+")
_expr_base = None
_REFS = ("refs", "", "")
_INFO = ("info", "", "")


def _expr():
    global _expr_base
    m = sys.modules.get("sqlglot.expressions.core")
    _expr_base = getattr(m, "Expr", None) if m else None


def fp(x, d=0, u=False):
    t = type(x)
    if t in _PRIM:
        # repr for numbers: hash(-1) == hash(-2), and 1 == 1.0 == True
        return hash((t.__name__, x if (t is str or t is bytes) else repr(x)))
    if d > 7:
        return hash(t.__qualname__)
    if t is tuple or t is list:
        if u:
            return hash((t.__name__, frozenset(_count([fp(i, d + 1, u) for i in x]))))
        return hash((t.__name__, tuple([fp(i, d + 1, u) for i in x])))
    if t is dict:
        if u:
            return hash(("dict", frozenset([(fp(k, d + 1, u), fp(v, d + 1, u)) for k, v in x.items()])))
        return hash(("dict", tuple([(fp(k, d + 1, u), fp(v, d + 1, u)) for k, v in x.items()])))
    if t is set or t is frozenset:
        return hash((t.__name__, frozenset([fp(i, d + 1, u) for i in x])))
    if isinstance(x, type):
        return hash(("class", x.__module__, x.__qualname__))
    if t is types.FunctionType:
        cells = []
        for c in x.__closure__ or ():
            try:
                cells.append(fp(c.cell_contents, d + 1, u))
            except ValueError:
                cells.append(0)
        # defaults are tracked under their own key (see snapshot): a changed default points at one function
        return hash(("func", x.__module__, x.__qualname__, tuple(cells)))
    if t is types.ModuleType:
        return hash(("module", x.__name__))
    if t is staticmethod or t is classmethod:
        return hash((t.__name__, fp(x.__func__, d + 1, u)))
    if t is property:
        return hash(("property", fp(x.fget, d + 1, u), fp(x.fset, d + 1, u)))
    if t is functools.partial:
        return hash(("partial", fp(x.func, d + 1, u), fp(x.args, d + 1, u), fp(x.keywords, d + 1, u)))
    if t is types.MethodType:
        return hash(("method", fp(x.__func__, d + 1, u), fp(type(x.__self__), d + 1)))
    if isinstance(x, enum.Enum):
        return hash(("enum", t.__module__, t.__qualname__, x.name, fp(x._value_, d + 1, u)))
    if t is re.Pattern:
        return hash(("re", x.pattern, x.flags))
    if _expr_base is not None and isinstance(x, _expr_base):
        try:
            return hash(("expr", t.__qualname__, repr(x)))
        except Exception:
            return hash(("expr", t.__qualname__))
    if isinstance(x, logging.Logger):
        return hash(("logger", x.name))
    dd = getattr(x, "__dict__", None)
    if type(dd) is dict and (t.__module__ or "").startswith("sqlglot"):
        return hash(("obj", t.__module__, t.__qualname__, fp(dd, d + 1, u)))
    try:
        return hash(("repr", t.__module__, t.__qualname__, _ADDR.sub("", repr(x))))
    except Exception:
        return hash(("opaque", t.__module__, t.__qualname__))


def _count(items):
    c = {}
    for i in items:
        c[i] = c.get(i, 0) + 1
    return c.items()


def snapshot(unstable=()):
    """{("g", module, name) | ("c", "module:qualname", attr): fingerprint} for every loaded sqlglot module.
    Keys in `unstable` hold containers whose order comes from iterating a set of classes, which differs from
    one import to the next on the pinned commit itself; those are compared without order."""
    _expr()
    out = {}
    seen = set()
    refs = set()
    info = {}

    def note_function(owner, name, v):
        f = v
        if isinstance(f, (staticmethod, classmethod)):
            f = f.__func__
        elif isinstance(f, property):
            f = f.fget
        if type(f) is types.FunctionType:
            key = ("d", owner, name)
            out[key] = hash((fp(f.__defaults__), fp(f.__kwdefaults__)))
            code = f.__code__
            info[key] = (code.co_filename, sorted({l for _, _, l in code.co_lines() if l is not None}))

    def visit_class(c):
        if c in seen:
            return
        seen.add(c)
        key = f"{c.__module__}:{c.__qualname__}"
        out[("c", key, "__bases__")] = fp(c.__bases__)
        for k, v in list(vars(c).items()):
            if k in _SKIP_CLASS:
                continue
            out[("c", key, k)] = fp(v, 0, ("c", key, k) in unstable)
            note_function(key, k, v)
            if isinstance(v, type) and v.__module__ == c.__module__ and v.__qualname__.startswith(c.__qualname__ + "."):
                visit_class(v)

    for name, m in list(sys.modules.items()):
        if m is None or not (name == "sqlglot" or name.startswith("sqlglot.")):
            continue
        for k, v in list(vars(m).items()):
            if k.startswith("__") and k != "__all__":
                continue
            out[("g", name, k)] = fp(v, 0, ("g", name, k) in unstable)
            if getattr(v, "__module__", None) == name:
                note_function(name, k, v)
            if isinstance(v, (type, types.ModuleType, types.FunctionType)):
                refs.add(("g", name, k))
            if isinstance(v, type) and v.__module__ == name:
                visit_class(v)
    out[_REFS] = refs
    out[_INFO] = info
    return out


def diff(a, b, lazy=()):
    """Keys whose fingerprint differs. A key in `lazy` (a module global that the pinned commit itself fills in
    on first use) is ignored when it exists on one side only."""
    out = []
    for k in a.keys() | b.keys():
        if k == _REFS or k == _INFO:
            continue
        x, y = a.get(k), b.get(k)
        if x == y:
            continue
        if (x is None or y is None) and k in lazy:
            continue
        out.append(k)
    return out
