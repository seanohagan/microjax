import inspect
from typing import Optional, List, Union, Sequence, Callable
from pprint import pformat
from copy import deepcopy
import llvmlite.ir as llvm_ir
import llvmlite.binding as llvm
from ctypes import CFUNCTYPE, c_float, Structure


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
        self._executor: Optional[Callable] = None
        self.vmapped = False
        self.jitted = False

    def __repr__(self):
        return pformat(self.__dict__)

    def __call__(self, *args):
        if self._executor:
            return self._executor(*args)

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


def jit(irf):
    if not isinstance(irf, IRF):
        irf = make_ir(irf)

    SIMD_WIDTH = 4

    module = llvm_ir.Module(name="microjax")
    data_type = (
        llvm_ir.FloatType()
        if not irf.vmapped
        else llvm_ir.VectorType(llvm_ir.FloatType(), SIMD_WIDTH)
    )
    ret_type = (
        data_type
        if len(irf.outputs) == 1
        else llvm_ir.ArrayType(data_type, len(irf.outputs))
    )
    ft = llvm_ir.FunctionType(
        ret_type,
        [data_type for _ in irf.inputs],
    )
    func = llvm_ir.Function(module, ft, name="func")

    symbols = {}

    for arg, ssa_arg in zip(func.args, irf.inputs):
        arg.name = ssa_arg
        symbols[ssa_arg] = arg

    block = func.append_basic_block(name="entry")

    builder = llvm_ir.IRBuilder(block)
    ops = {"add": builder.fadd, "mul": builder.fmul}
    for instruction in irf.instructions:
        var_name, op, operand1, operand2 = instruction

        op1 = symbols.get(operand1, llvm_ir.Constant(data_type, operand1))
        op2 = symbols.get(operand2, llvm_ir.Constant(data_type, operand2))

        results = ops[op](op1, op2, name=var_name)
        symbols[var_name] = results

    if len(irf.outputs) == 1:
        builder.ret(symbols[irf.outputs[0]])
    else:
        values_to_return = [
            symbols.get(output, llvm_ir.Constant(llvm_ir.FloatType(), 0.0))
            for output in irf.outputs
        ]
        array_ptr = builder.alloca(ret_type)
        for idx, val in enumerate(values_to_return):
            index_constant = llvm_ir.Constant(llvm_ir.IntType(32), idx)
            elem_ptr = builder.gep(
                array_ptr, [llvm_ir.Constant(llvm_ir.IntType(32), 0), index_constant]
            )
            builder.store(val, elem_ptr)

        loaded_array = builder.load(array_ptr)
        builder.ret(loaded_array)

    llvm.initialize()
    llvm.initialize_native_target()
    llvm.initialize_all_asmprinters()
    module.triple = "arm64-apple-darwin21.5.0"
    module.data_layout = "e-m:o-i64:64-i128:128-n32:64-S128"
    print(str(module))

    with open("output.ll", "w") as f:
        f.write(str(module))

    target = llvm.Target.from_default_triple()
    target_machine = target.create_target_machine()
    compiled_engine = llvm.create_mcjit_compiler(
        llvm.parse_assembly(str(module)), target_machine
    )
    c_data_type = c_float if not irf.vmapped else c_float * SIMD_WIDTH
    if len(irf.outputs) == 1:
        c_ret_type = c_data_type
    else:

        class ReturnStruct(Structure):
            _fields_ = [(f"f{i+1}", c_data_type) for i, _ in enumerate(irf.outputs)]

        c_ret_type = ReturnStruct
    cfunctype = CFUNCTYPE(c_ret_type, *(c_data_type for _ in irf.inputs))
    jf = JittedFunc(
        func,
        cfunctype,
        module,
        target,
        target_machine,
        compiled_engine,
        irf.vmapped,
        SIMD_WIDTH,
    )

    new_irf = deepcopy(irf)
    new_irf._executor = jf
    return new_irf


class JittedFunc:
    def __init__(
        self, func, cfunctype, module, target, target_machine, engine, vmapped, simdw
    ):
        self.module = module
        self.func = func
        self.target = target
        self.target_machine = target_machine
        self.engine = engine
        self.ptr = self.engine.get_function_address("func")
        self.cfunc = cfunctype(self.ptr)
        self.vmapped = vmapped
        self.simdw = simdw

    def __call__(self, *args):
        if not self.vmapped:
            result = self.cfunc(*args)
            if isinstance(result, Structure):
                return tuple(
                    getattr(result, f"f{i+1}") for i in range(len(result._fields_))
                )
            return result
        else:
            print(args[0])
            dtype = c_float * 4
            inputdata = dtype(*args[0])
            print(inputdata, list(inputdata))
            return list(self.cfunc(inputdata))
        #     q, r = divmod(len(args[0]), self.simdw)
        #     ret = []
        #     for i in range(q):
        #         batch_args = [
        #             arg[(i * self.simdw) * ((i + 1) * self.simdw)] for arg in args
        #         ]
        #         batch_res = self.cfunc


def vmap(irf):
    if not isinstance(irf, IRF):
        irf = make_ir(irf)

    def func(
        argsl: Union[Sequence[Union[int, float]], Sequence[tuple]]
    ) -> List[Union[float, tuple]]:
        if all(isinstance(arg, (int, float)) for arg in argsl):
            return [irf(argset) for argset in argsl]
        elif all(isinstance(arg, tuple) for arg in argsl):
            return [irf(*argset) for argset in argsl]  # type: ignore
        else:
            raise ValueError("Input is of wrong type")

    new_irf = deepcopy(irf)
    new_irf.vmapped = True
    new_irf._executor = func
    return new_irf
