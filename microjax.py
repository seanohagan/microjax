import inspect


def base26(n):
    s = ""
    while n >= 0:
        s = chr(n % 26 + ord("a")) + s
        n = n // 26 - 1
    return "$" + s


# TODO:get constants to work in tracing
class Tracer:
    def __init__(self, value, trace):
        self.value = value
        self.trace = trace

    def _ensure_tracer(self, other):
        if isinstance(other, Tracer):
            return other.value
        return str(other)

    def __add__(self, other):
        _other = self._ensure_tracer(other)
        b26n = base26(len(self.trace))
        result = Tracer(b26n, self.trace)
        self.trace.append([b26n, "add", self.value, _other])
        return result

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        _other = self._ensure_tracer(other)
        b26n = base26(len(self.trace))
        result = Tracer(b26n, self.trace)
        self.trace.append([b26n, "mul", self.value, _other])
        return result

    def __rmul__(self, other):
        return self * other


def get_ir(func):
    trace = []
    arg_names = inspect.getfullargspec(func).args
    inputs = [Tracer(arg_name, trace) for arg_name in arg_names]
    func(*inputs)
    return trace


def backprop(ir):
    # Initialize the gradient of the last computation to 1
    grads = {ir[-1][0]: 1}
    grad_ir = []

    def add_grad(out_grad, operand1, operand2):
        return [out_grad, out_grad]
    def mul_grad(out_grad, operand1, operand2):
        return [out_grad * operand2, out_grad * operand1]
    grad_ops = {
        "add": add_grad,
        "mul": mul_grad,
    }
    return grad_ir


ir = [
    ["mul", "x", "y"],
    ["add", 0, 45],
    ["add", "z", -2],
    ["add", "y", 2],
    ["mul", 1, 3],
    ["add", 4, "z"],
    ["mul", "x", -2],
    ["add", 5, 6],
    ["add", 7, "y"],
]
# print(backprop(ir))


def f(x, y):
    a = x + y
    b = a * y
    return b


def g(x, y):
    return 4 * (x * y + x)


def h(x, y, z):
    a = x * y + 45
    d = z + -2
    b = a * (y + d)
    c = b + z + (-2 * x)
    return c + y


print(get_ir(f))
print(get_ir(g))
print(get_ir(h))
