import inspect


class Operand:
    def __init__(self, value, is_symbolic=True):
        self.value = value
        self.is_symbolic = is_symbolic

    def __add__(self, other):
        if isinstance(other, (int, float)):
            return Operand(self.value + other)
        elif isinstance(other, Operand):
            return Operand(self.value + other.value)
        else:
            raise ValueError(f"Cannot add Operand with {type(other)}")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Operand(self.value * other)
        elif isinstance(other, Operand):
            return Operand(self.value * other.value)
        else:
            raise ValueError(f"Cannot multiply Operand with {type(other)}")

    def __radd__(self, other):  # for situations like int + Operand
        return self + other

    def __rmul__(self, other):  # for situations like int * Operand
        return self * other

    def __repr__(self):
        return f"Operand({self.value}, is_symbolic={self.is_symbolic})"


class Tracer:
    def __init__(self, name, trace):
        self.name = Operand(name)
        self.trace = trace

    def _binary_op(self, other, op):
        if isinstance(other, Tracer):
            _other = other.name
        else:
            _other = Operand(other, is_symbolic=False)

        operation = [op, self.name, _other]
        self.trace.append(operation)
        return Tracer(len(self.trace), self.trace)

    def __add__(self, other):
        return self._binary_op(other, "add")

    def __mul__(self, other):
        return self._binary_op(other, "mul")

    def __repr__(self):
        return str(self.name)


def get_ir(func):
    trace = []
    params = list(inspect.signature(func).parameters.keys())
    tracers = {p: Tracer(p, trace) for p in params}
    func(**tracers)
    return trace


def f(x, y):
    a = x + y
    b = a * y
    return b


def g(x, y, z, v45, vm2):
    a = x * y + v45
    d = z + vm2
    b = a * (y + d)
    c = b + z + (vm2 * x)
    return c + y


def h(x, y, z):
    a = x * y + 45
    d = z + -2
    b = a * (y + d)
    c = b + z + (-2 * x)
    return c + y


print(get_ir(f))
print(get_ir(g))
print(get_ir(h))
