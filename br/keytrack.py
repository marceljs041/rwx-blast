"""Which entries of class-level dict tables a test reads.

Build time: every dict that only class namespaces refer to is replaced by a TDict with the same content. A TDict
notes each key it is asked for, and notes "all of it" for every other operation. Run time: when a re-import
shows that such a table changed, the pinned and the patched table are compared entry by entry; only a test that
asked for a changed entry, or looked at the table as a whole, can see the change.

Everything here describes the pinned commit: which test asks which table for which key."""
import enum
import gc
import hashlib
import sys
import types

from br import statediff

READS = set()    # (table id, key asked for), present or not
WHOLE = set()    # table ids that were iterated, copied, compared, printed or modified
LENS = set()     # table ids whose size or truth value was taken

_KEY = {"__getitem__", "get", "__contains__"}
_SIZE = {"__len__"}
_UNTOUCHED = {"__new__", "__init__", "__hash__", "__getattribute__", "__sizeof__", "__class_getitem__", "__doc__",
              "fromkeys"}
_ALL = {"__repr__", "__lt__", "__le__", "__eq__", "__ne__", "__gt__", "__ge__", "__iter__", "__or__", "__ror__",
        "__ior__", "__setitem__", "__delitem__", "setdefault", "pop", "popitem", "keys", "items", "values",
        "update", "clear", "copy", "__reversed__"}
# A dict method this file does not know would go unnoticed, so tracking is refused on such a Python.
SUPPORTED = set(dict.__dict__) <= (_KEY | _SIZE | _UNTOUCHED | _ALL)


class TDict(dict):
    __slots__ = ("_br",)

    def __getitem__(self, k):
        READS.add((self._br, k))
        return dict.__getitem__(self, k)

    def get(self, k, default=None):
        READS.add((self._br, k))
        return dict.get(self, k, default)

    def __contains__(self, k):
        READS.add((self._br, k))
        return dict.__contains__(self, k)

    def __len__(self):
        LENS.add(self._br)
        return dict.__len__(self)

    def __reduce_ex__(self, protocol):
        # copies and pickles are plain dicts
        WHOLE.add(self._br)
        return dict, (dict.copy(self),)

    def __reduce__(self):
        WHOLE.add(self._br)
        return dict, (dict.copy(self),)


def _all(name):
    base = getattr(dict, name)

    def method(self, *args, **kwargs):
        WHOLE.add(self._br)
        return base(self, *args, **kwargs)

    method.__name__ = name
    return method


# Overriding __iter__ also keeps C code (dict(d), {**d}, f(**d), x.update(d)) off its fast path for exact
# dicts: it then asks for keys() and each item, which is noted.
for _name in _ALL:
    setattr(TDict, _name, _all(_name))


def reset():
    READS.clear()
    WHOLE.clear()
    LENS.clear()


def tok(k):
    """A name for a dict key that is the same in every process and after a re-import, or None. Keys that are
    equal as dict keys get the same name."""
    if isinstance(k, str):
        return "s" + str.__str__(k)
    if k is None:
        return "N"
    if isinstance(k, (bool, int, float)):
        try:
            if k == int(k):
                return "n" + repr(int(k))
        except (ValueError, OverflowError):
            return None
        return "n" + repr(float(k))
    if isinstance(k, enum.Enum):
        t = type(k)
        return f"e{t.__module__}:{t.__qualname__}.{k.name}"
    if isinstance(k, type):
        return f"c{k.__module__}:{k.__qualname__}"
    if type(k) is tuple:
        parts = [tok(i) for i in k]
        if None in parts:
            return None
        return "t" + str(len(parts)) + "\x1f" + "\x1f".join(parts)
    if type(k) is bytes:
        return "b" + k.hex()
    return None


def digest(token):
    return int.from_bytes(hashlib.blake2b(token.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "little")


def dump():
    """What the running test read so far: {"keys": {table: [key digests]}, "whole": [tables], "len": [tables]}."""
    whole = set(WHOLE)
    keys = {}
    for tid, k in READS:
        t = tok(k)
        if t is None:
            whole.add(tid)
        else:
            keys.setdefault(tid, set()).add(digest(t))
    return dict(keys={t: sorted(v) for t, v in keys.items() if t not in whole},
                whole=sorted(whole), len=sorted(LENS - whole))


def classes(modules=None):
    """Every class defined in a loaded sqlglot module, nested ones included (the walk statediff.snapshot does)."""
    seen = set()
    out = []

    def visit(c):
        if c in seen:
            return
        seen.add(c)
        out.append(c)
        for v in list(vars(c).values()):
            if isinstance(v, type) and v.__module__ == c.__module__ and v.__qualname__.startswith(c.__qualname__ + "."):
                visit(v)

    for name, m in list((modules or sys.modules).items()):
        if m is None or not (name == "sqlglot" or name.startswith("sqlglot.")):
            continue
        for v in list(vars(m).values()):
            if isinstance(v, type) and v.__module__ == name:
                visit(v)
    return out


def _only_in_namespaces(d, own, objs, spaces):
    """Is `d` referred to by nothing but the class attributes and module globals listed in `own`? (No nested
    function here: a closure would hold `d` in a cell and count as a referrer itself.)"""
    for r in gc.get_referrers(d):
        if r is objs or isinstance(r, types.FrameType):
            continue
        if type(r) is not dict or id(r) not in spaces:
            return False
        for k, v in r.items():
            if v is d and (id(r), k) not in own:
                return False
    return True


def _only_in(v, parent, objs):
    for r in gc.get_referrers(v):
        if r is parent or r is objs or isinstance(r, types.FrameType):
            continue
        return False
    return True


def install():
    """Replaces dict tables by tracking copies. Returns {state key: table id}, where a state key is what
    statediff uses, ("c", class, attribute) or ("g", module, name), or such a key extended by a key digest for
    a dict that sits one level down inside a module-level dict. A dict that anything but class and module
    namespaces refers to stays as it is: a reference taken earlier would bypass the copy."""
    if not SUPPORTED:
        return {}
    gc.collect()
    owners = {}      # id(dict) -> [(holder, attribute, state key)]
    objs = {}
    spaces = set()   # ids of the namespaces looked at
    count = {}
    for name, m in list(sys.modules.items()):
        if m is None or not (name == "sqlglot" or name.startswith("sqlglot.")) or type(m) is not types.ModuleType:
            continue
        ns = vars(m)
        spaces.add(id(ns))
        for k, v in ns.items():
            if type(v) is dict and not k.startswith("__"):
                owners.setdefault(id(v), []).append((m, k, ("g", name, k), id(ns)))
                objs[id(v)] = v
    for c in classes():
        key = f"{c.__module__}:{c.__qualname__}"
        count[key] = count.get(key, 0) + 1
        if issubclass(c, enum.Enum):
            continue
        ns = [r for r in gc.get_referents(c.__dict__) if type(r) is dict]
        if len(ns) != 1:
            continue
        spaces.add(id(ns[0]))
        for k, v in ns[0].items():
            if type(v) is dict and not k.startswith("__"):
                owners.setdefault(id(v), []).append((c, k, ("c", key, k), id(ns[0])))
                objs[id(v)] = v
    table = {}
    next_id = 0
    for ident, own in owners.items():
        d = objs[ident]
        if not _only_in_namespaces(d, {(space, k) for _, k, _, space in own}, objs, spaces):
            continue
        if [1 for _, _, key, _ in own if key[0] == "c" and count[key[1]] != 1]:
            continue                 # two classes share a name: the state key would not say which one
        nested = {}
        if [1 for _, _, key, _ in own if key[0] == "g"]:
            # a module-level dict of dicts (a registry or cache of tables): track the inner tables too
            for k in list(d):        # not items(): its iterator keeps the last pair in a tuple of its own
                v = d[k]
                name = tok(k) if type(v) is dict else None
                if name is not None and id(v) not in objs and _only_in(v, d, objs):
                    nested[k] = digest(name)
        t = TDict(d)                 # made after the check above: the copy refers to the inner dicts too
        t._br = next_id
        done = []
        try:
            for holder, k, _, _ in own:
                setattr(holder, k, t)
                done.append((holder, k))
        except Exception:
            for holder, k in done:
                (type.__setattr__ if isinstance(holder, type) else setattr)(holder, k, d)
            continue
        for _, _, key, _ in own:
            table[key] = next_id
        next_id += 1
        for k, h in nested.items():
            inner = TDict(d[k])
            inner._br = next_id
            dict.__setitem__(t, k, inner)
            for _, _, key, _ in own:
                table[key + (h,)] = next_id
            next_id += 1
    return table


# ---------------------------------------------------------------- run time

MISSING = object()


def lookup(modules, key):
    """The value a state key ("c", "module:qualname", attribute) or ("g", module, name) stands for."""
    kind, owner, name = key
    if kind == "g":
        return vars(modules[owner])[name]
    mod, _, qual = owner.partition(":")
    obj = modules[mod]
    for part in qual.split("."):
        obj = vars(obj)[part]
    return vars(obj)[name]


def entries(d):
    """{key digest: value}, or None when a key has no stable name."""
    out = {}
    for k, v in d.items():
        t = tok(k)
        if t is None:
            return None
        h = digest(t)
        if h in out:
            return None
        out[h] = v
    return out


def changed_entries(old, new, old_expr, unordered=False):
    """({key digest: (pinned value, patched value)} for the entries that differ, whether the size differs), or
    None when the two values cannot be compared entry by entry. A side without the entry holds MISSING.
    `old_expr` is the pinned import's expression base class: fingerprints of expression objects depend on it."""
    if type(old) is not dict or type(new) is not dict:
        return None
    a, b = entries(old), entries(new)
    if a is None or b is None:
        return None
    fb = {h: statediff.fp(v, 1, unordered) for h, v in b.items()}
    keep = statediff._expr_base
    statediff._expr_base = old_expr
    try:
        fa = {h: statediff.fp(v, 1, unordered) for h, v in a.items()}
    finally:
        statediff._expr_base = keep
    out = {h: (a.get(h, MISSING), b.get(h, MISSING)) for h in fa.keys() | fb.keys() if fa.get(h) != fb.get(h)}
    return out, len(old) != len(new)
