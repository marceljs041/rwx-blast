"""Statement units: lets a test run only those of its top-level check statements that can see a change.

A test method is rewritten so that every top-level statement of the form `self.validate...(...)` or
`self.assert...(...)` runs under `if _br_begin(k):`. At build time `_br_begin` always answers yes and marks the
unit for the tracer; at run time it answers no for units that execute no changed statement and use no changed
class. Only undecorated test methods are rewritten, and a test whose result changes when half of its units are
left out (its statements depend on each other) is recorded at build time and never has units skipped."""
import ast
import builtins
import sys
import types


class UnitCtx:
    affected = None   # units that must run; None = all
    known = None      # number of units the build saw for the running test; a unit past it always runs
    hook = None       # build time: called with (0|1, k)
    active = False    # a unit is running right now
    next = 0          # number given to the next subTest block of the running test


SU = UnitCtx()


def _br_begin(k):
    h = SU.hook
    if h is not None:
        h(0, k)
        SU.active = True
        return True
    a = SU.affected
    if a is None or k in a or (SU.known is not None and k >= SU.known):
        SU.active = True
        return True
    return False


def _br_reset(first):
    SU.next = first


def _br_next():
    k = SU.next
    SU.next = k + 1
    return k


def _br_end(k):
    SU.active = False
    h = SU.hook
    if h is not None:
        h(1, k)


builtins._br_begin = _br_begin
builtins._br_end = _br_end
builtins._br_reset = _br_reset
builtins._br_next = _br_next

_BAD = (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)


def _skippable(stmt):
    if not (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)):
        return False
    f = stmt.value.func
    if not (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name) and f.value.id == "self"):
        return False
    if not (f.attr.startswith("validate") or f.attr.startswith("assert")):
        return False
    return not any(isinstance(n, _BAD) for n in ast.walk(stmt))


def _is_subtest(stmt):
    return isinstance(stmt, ast.With) and any(
        isinstance(i.context_expr, ast.Call) and isinstance(i.context_expr.func, ast.Attribute)
        and isinstance(i.context_expr.func.value, ast.Name) and i.context_expr.func.value.id == "self"
        and i.context_expr.func.attr == "subTest" for i in stmt.items)


def _guard_subtests(stmts, counter):
    """Inside a loop: every `with self.subTest(...)` block becomes a unit numbered in the order the blocks
    run (after the fixed units). Returns the new statement list."""
    out = []
    for st in stmts:
        if _is_subtest(st) and not any(isinstance(n, _BAD) for n in ast.walk(st)):
            name = f"_br_k{counter[0]}"
            counter[0] += 1
            out.append(ast.copy_location(ast.Assign(
                targets=[ast.Name(id=name, ctx=ast.Store())],
                value=ast.Call(func=ast.Name(id="_br_next", ctx=ast.Load()), args=[], keywords=[])), st))
            call = lambda f: ast.Call(func=ast.Name(id=f, ctx=ast.Load()), args=[ast.Name(id=name, ctx=ast.Load())],
                                      keywords=[])
            out.append(ast.copy_location(ast.If(test=call("_br_begin"), body=[ast.Try(
                body=[st], handlers=[], orelse=[], finalbody=[ast.Expr(call("_br_end"))])], orelse=[]), st))
            continue
        if isinstance(st, (ast.For, ast.While)):
            st.body = _guard_subtests(st.body, counter)
            st.orelse = _guard_subtests(st.orelse, counter)
        elif isinstance(st, ast.If):
            st.body = _guard_subtests(st.body, counter)
            st.orelse = _guard_subtests(st.orelse, counter)
        elif isinstance(st, (ast.With, ast.Try)) and not _is_subtest(st):
            st.body = _guard_subtests(st.body, counter)
        out.append(st)
    return out


def _rewrite(func):
    """Returns (new FunctionDef, number of fixed units, number of subTest blocks made units)."""
    body = []
    k = 0
    counter = [0]
    for stmt in func.body:
        if isinstance(stmt, (ast.For, ast.While)):
            stmt.body = _guard_subtests(stmt.body, counter)
            body.append(stmt)
            continue
        if _skippable(stmt):
            call = lambda name: ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=[ast.Constant(k)], keywords=[])
            guarded = ast.If(
                test=call("_br_begin"),
                body=[ast.Try(body=[stmt], handlers=[], orelse=[], finalbody=[ast.Expr(call("_br_end"))])],
                orelse=[],
            )
            ast.copy_location(guarded, stmt)
            body.append(guarded)
            k += 1
        else:
            body.append(stmt)
    if counter[0]:
        reset = ast.Expr(ast.Call(func=ast.Name(id="_br_reset", ctx=ast.Load()), args=[ast.Constant(k)], keywords=[]))
        body.insert(0, ast.copy_location(reset, func.body[0]))
    new = ast.FunctionDef(name=func.name, args=func.args, body=body, decorator_list=[], returns=None,
                          type_comment=None, type_params=[])
    ast.copy_location(new, func)
    return new, k, counter[0]


def _find_code(code, name):
    for c in code.co_consts:
        if isinstance(c, types.CodeType) and c.co_name == name:
            return c
    return None


def instrument(module):
    """Rewrites the undecorated test methods of the TestCase classes defined in `module`."""
    if getattr(module, "__br_done__", False):
        return
    module.__br_done__ = True
    path = getattr(module, "__file__", None)
    if not path or not path.endswith(".py"):
        return
    try:
        with open(path, encoding="utf-8") as f:
            tree = ast.parse(f.read())
    except Exception:
        return
    for cnode in tree.body:
        if not isinstance(cnode, ast.ClassDef):
            continue
        cls = vars(module).get(cnode.name)
        if not isinstance(cls, type) or cls.__module__ != module.__name__:
            continue
        for fnode in cnode.body:
            if not isinstance(fnode, ast.FunctionDef) or not fnode.name.startswith("test") or fnode.decorator_list:
                continue
            fn = vars(cls).get(fnode.name)
            if type(fn) is not types.FunctionType or fn.__code__.co_firstlineno != fnode.lineno:
                continue
            new, n, dyn = _rewrite(fnode)
            if n < 2 and not dyn:
                continue
            wrapper = ast.Module(body=[ast.ClassDef(name=cnode.name, bases=[], keywords=[], body=[new],
                                                    decorator_list=[], type_params=[])], type_ignores=[])
            ast.fix_missing_locations(wrapper)
            try:
                mod_code = compile(wrapper, path, "exec", dont_inherit=True)
                code = _find_code(_find_code(mod_code, cnode.name), fnode.name)
            except Exception:
                continue
            old = fn.__code__
            if code is None or code.co_freevars != old.co_freevars or code.co_argcount != old.co_argcount:
                continue
            fn.__code__ = code
            fn.__br_units__ = n
