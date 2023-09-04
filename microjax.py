import inspect
from pprint import pformat
from copy import deepcopy


def base26(n):
    s = ""
    while n >= 0:
        s = chr(n % 26 + ord("a")) + s
        n = n // 26 - 1
    return "$" + s


def evaluate_ir(ir, subs):
    environment = {}
    operations = {
        "add": lambda x, y: x + y,
        "mul": lambda x, y: x * y,
    }
    for instruction in ir:
        var_name, op, operand1, operand2 = instruction

        operand1 = subs.get(operand1, operand1)
        operand2 = subs.get(operand2, operand2)

        op1_val = environment.get(operand1, operand1)
        op2_val = environment.get(operand2, operand2)

        environment[var_name] = operations[op](op1_val, op2_val)

    return environment


class IRF:
    def __init__(self, instructions, inputs, outputs):
        self.instructions = instructions
        self.inputs = inputs
        self.outputs = outputs

    def __repr__(self):
        return pformat(self.__dict__)

    def __call__(self, *args):
        substitutions = dict(zip(self.inputs, args))
        values = evaluate_ir(self.instructions, substitutions)
        return (
            values[self.outputs[0]]
            if len(self.outputs) == 1
            else tuple(values.get(var_name, 0) for var_name in self.outputs)
        )


class Tracer:
    def __init__(self, value, trace):
        self.value = value
        self.trace = trace

    def _ensure_tracer(self, other):
        if isinstance(other, Tracer):
            return other.value
        return other

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


def make_ir(func):
    trace = []
    arg_names = inspect.getfullargspec(func).args
    inputs = [Tracer(arg_name, trace) for arg_name in arg_names]
    outputs = func(*inputs)
    if isinstance(outputs, (tuple, list)):
        output_names = [output.value for output in outputs]
    else:
        output_names = [outputs.value]
    return IRF(trace, arg_names, output_names)


def grad(f):
    if not isinstance(f, IRF):
        f = make_ir(f)

    ir = f.instructions

    grad_ir = deepcopy(ir)
    grads = {
        output: {inter: int(inter == output) for inter in [instr[0] for instr in ir]}
        for output in f.outputs
    }
    influences = {
        output: {inter: inter == output for inter in [instr[0] for instr in ir]}
        for output in f.outputs
    }

    def add_grad(operand1, operand2):
        return [1, 1]

    def mul_grad(operand1, operand2):
        return [operand2, operand1]

    grad_ops = {
        "add": add_grad,
        "mul": mul_grad,
    }

    for i, operation in enumerate(reversed(ir)):
        res, op, op1, op2 = operation
        grad_ops_out = grad_ops[op](op1, op2)

        for output in f.outputs:
            if influences[output][res]:
                res_ref = (
                    f"{output}/{res}#{grads[output].get(res, 0)}"
                    if res != output
                    else 1
                )

                if isinstance(op1, str):
                    grads[output][op1] = grads[output].get(op1, 0) + 1
                    grad_name = f"{output}/{op1}#{grads[output].get(op1, 0)}"
                    if grads[output].get(op1, 0) == 1:
                        grad_ir.append([grad_name, "mul", res_ref, grad_ops_out[0]])
                    else:
                        prev_name = f"{output}/{op1}#{grads[output].get(op1, 0)-1}"
                        inc_name = prev_name + "#inc"
                        grad_ir.append([inc_name, "mul", res_ref, grad_ops_out[0]])
                        grad_ir.append([grad_name, "add", prev_name, inc_name])

                    influences[output][op1] = True

                if isinstance(op2, str):
                    grads[output][op2] = grads[output].get(op2, 0) + 1
                    grad_name = f"{output}/{op2}#{grads[output].get(op2, 0)}"
                    if grads[output].get(op2, 0) == 1:
                        grad_ir.append([grad_name, "mul", res_ref, grad_ops_out[1]])
                    else:
                        prev_name = f"{output}/{op2}#{grads[output].get(op2, 0)-1}"
                        inc_name = prev_name + "#inc"
                        grad_ir.append([inc_name, "mul", res_ref, grad_ops_out[1]])
                        grad_ir.append([grad_name, "add", prev_name, inc_name])

                    influences[output][op2] = True

    outputs = [
        f"{output}/{input}#{grads[output].get(input,0)}"
        for input in f.inputs
        for output in f.outputs
    ]

    out_irf = IRF(grad_ir, f.inputs, outputs)
    return out_irf


#
# def f(x, y):
#     a = x + y
#     b = a * y
#     return b
#
#
def g(x, y):
    return x + 2, y * 3


print(make_ir(g))

print(grad(g))

#
#
# def h(x, y, z):
#     a = x * y + 45
#     d = z + -2
#     b = a * (y + d)
#     c = b + z + (-2 * x)
#     return c + y
#
#
# g_ir = get_ir(g)
# print(g_ir.inputs)
# print(g_ir.outputs)
# print(g_ir.instructions)
# print(g(4, 3))
# print(g_ir(4, 3))
#
# print("grad_ir:")
# print("g: 4xy + 4x\ngrad g: (4y+4, 4x)")
grad_g = grad(g)
print(grad_g)
print(grad_g(3, 4))
hess_g = grad(grad_g)
print(hess_g)
print(hess_g(3, 4))


# print(grad_g_ir.inputs)
# print(grad_g_ir.outputs)
# print(grad_g_ir.instructions)
# print(grad_g_ir(4, 3))
# print(grad_g_ir(11, 4))
#
def morg(x, y, z):
    return 3 * x + y * y + z * z * z


grad_morg = grad(morg)
print(grad_morg)
# print(grad_morg.inputs)
# # print(grad_morg.outputs)
# # print(grad_morg.instructions)
# # print(evaluate_ir(grad_morg.instructions, {"x": 1, "y": 2, "z": 3}))
print(grad_morg(1, 2, 3))
print(grad_morg(4, 5, 6))

hess_morg = grad(grad_morg)
print(hess_morg)
print(hess_morg(1, 2, 3))
print(hess_morg(4, 5, 6))
