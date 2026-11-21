"""Old-versus-new evaluation of a swapped comparison operator.

When a patch changes exactly one comparison operator (`is` to `==`, `<` to `<=`, ...), the patched function
is compiled with that comparison routed through `_br_cmp`, which applies the new operator (the value the
patched code uses) and also the old one on the same two operands, and counts how often the two disagree.
A test in which they never disagree behaves exactly as it does on the pinned commit."""
import ast
import builtins
import enum
import operator


class Tally:
    count = 0       # times the changed comparison ran
    diverged = 0    # times old and new operator disagreed
    open_ended = 0  # times it ran on operands for which agreement so far says little about the next run


IDENTITY_SWAPS = {("Is", "Eq"), ("Eq", "Is"), ("IsNot", "NotEq"), ("NotEq", "IsNot")}


def _singleton(x):
    """None, True, False, a class or an enum member: `is` and `==` against one of these can only differ when
    the other operand has an unusual __eq__, not when it merely holds an unusual value."""
    return x is None or x is True or x is False or isinstance(x, (type, enum.Enum))


OPS = {
    "Is": operator.is_, "IsNot": operator.is_not, "Eq": operator.eq, "NotEq": operator.ne,
    "Lt": operator.lt, "LtE": operator.le, "Gt": operator.gt, "GtE": operator.ge,
    "In": lambda a, b: a in b, "NotIn": lambda a, b: a not in b,
}


def _br_cmp(left, right, old, new):
    value = OPS[new](left, right)
    Tally.count += 1
    if (old, new) not in IDENTITY_SWAPS or not (_singleton(left) or _singleton(right)):
        Tally.open_ended += 1
    try:
        if bool(OPS[old](left, right)) != bool(value):
            Tally.diverged += 1
    except Exception:
        Tally.diverged += 1
    return value


builtins._br_cmp = _br_cmp


def _diff(a, b):
    """The single pair of subtrees in which two ASTs differ, or None when they are equal. When they differ in
    more than one place the common parent is returned; a changed operator is reported as its Compare node."""
    if type(a) is not type(b):
        return a, b
    found = []
    for field in a._fields:
        x, y = getattr(a, field, None), getattr(b, field, None)
        if isinstance(x, list) and isinstance(y, list):
            if len(x) != len(y):
                return a, b
            for p, q in zip(x, y):
                if isinstance(p, ast.AST) and isinstance(q, ast.AST):
                    d = _diff(p, q)
                    if d:
                        found.append(d)
                elif p != q:
                    return a, b
        elif isinstance(x, ast.AST) and isinstance(y, ast.AST):
            d = _diff(x, y)
            if d:
                found.append(d)
        elif x != y:
            return a, b
    if not found:
        return None
    if len(found) > 1 or isinstance(found[0][0], ast.cmpop):
        return a, b
    return found[0]


class _Swap(ast.NodeTransformer):
    def __init__(self, target, replacement):
        self.target, self.replacement = target, replacement

    def visit(self, node):
        if node is self.target:
            return self.replacement
        return self.generic_visit(node)


def instrument(old_src, new_src, path):
    """The patched module compiled with its one changed comparison counted, or None when the patch is anything
    other than one swapped operator between the same two operands."""
    try:
        old_tree, new_tree = ast.parse(old_src), ast.parse(new_src)
    except SyntaxError:
        return None
    d = _diff(old_tree, new_tree)
    if not d:
        return None
    o, n = d
    if not (isinstance(o, ast.Compare) and isinstance(n, ast.Compare) and len(o.ops) == 1 and len(n.ops) == 1):
        return None
    old_op, new_op = type(o.ops[0]).__name__, type(n.ops[0]).__name__
    if old_op == new_op or old_op not in OPS or new_op not in OPS:
        return None
    if ast.dump(o.left) != ast.dump(n.left) or ast.dump(o.comparators[0]) != ast.dump(n.comparators[0]):
        return None
    for sub in ast.walk(n):
        if isinstance(sub, (ast.NamedExpr, ast.Yield, ast.YieldFrom, ast.Await)):
            return None
    call = ast.Call(func=ast.Name(id="_br_cmp", ctx=ast.Load()),
                    args=[n.left, n.comparators[0], ast.Constant(old_op), ast.Constant(new_op)], keywords=[])
    ast.copy_location(call, n)
    tree = _Swap(n, call).visit(new_tree)
    ast.fix_missing_locations(tree)
    try:
        return compile(tree, path, "exec", dont_inherit=True)
    except Exception:
        return None
