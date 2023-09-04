import inspect
from pprint import pformat
from copy import deepcopy


def base26(n):
    if not isinstance(n, int) or n < 0:
        raise ValueError("Input n must be a nonnegative integer")

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

    pruned_grad_ir = [
        instr
        for instr in grad_ir
        if instr not in f.instructions
        or any([influences[outvar].get(instr[0], False) for outvar in f.outputs])
    ]

    out_irf = IRF(pruned_grad_ir, f.inputs, outputs)

    return out_irf
