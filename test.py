from microjax import base26, evaluate_ir, IRF, make_ir, grad
import pytest


def test_base26():
    assert base26(0) == "$a"
    assert base26(1) == "$b"
    assert base26(25) == "$z"
    assert base26(27) == "$ab"
    assert base26(51) == "$az"
    assert base26(52) == "$ba"
    assert base26(702) == "$aaa"
    with pytest.raises(ValueError):
        base26(-1)


@pytest.mark.parametrize(
    "ir,subs,expected",
    [
        ([["a", "add", "x", "y"]], {"x": 1, "y": 2}, {"a": 3}),
        (
            [["a", "mul", "x", "y"], ["b", "add", "a", "z"]],
            {"x": 2, "y": 3, "z": 1},
            {"a": 6, "b": 7},
        ),
    ],
)
def test_evaluate_ir(ir, subs, expected):
    assert evaluate_ir(ir, subs) == expected


@pytest.mark.parametrize(
    "ir, inputs, outputs, args, expected",
    [
        ([["a", "add", "x", "y"]], ["x", "y"], ["a"], (1, 2), 3),
        (
            [["a", "add", "x", "y"], ["b", "mul", "x", "y"]],
            ["x", "y"],
            ["a", "b"],
            (2, 3),
            (5, 6),
        ),
    ],
)
def test_IRF_call(ir, inputs, outputs, args, expected):
    func = IRF(ir, inputs, outputs)
    assert func(*args) == expected


def test_make_ir_single_input_single_output():
    def f(x):
        return x + 2

    irf = make_ir(f)

    assert irf.inputs == ["x"]
    assert irf.outputs == ["$a"]
    assert irf(5) == 7
    assert irf(0) == 2


def test_make_ir_multi_input_single_output():
    def f(x, y):
        return 2 * x + 7.0 * y * y + 1

    irf = make_ir(f)

    assert irf.inputs == ["x", "y"]
    assert irf(3, 4) == 119
    assert irf(0, 0) == 1


def test_make_ir_multi_input_multi_output():
    def f(x, y, z):
        return x + y + (-1 * z), x * y * z

    irf = make_ir(f)

    assert irf.inputs == ["x", "y", "z"]
    assert irf(2, 3, 1) == (4, 6)


def test_grad_single_input_single_output():
    def func(x):
        return x * x

    ir = make_ir(func)
    grad_f = grad(ir)
    assert grad_f(2) == 4
    assert grad_f(17) == 34


def test_grad_multi_input_single_output():
    def func(x, y):
        return x * y + y

    ir = make_ir(func)
    grad_f = grad(ir)
    assert grad_f(2, 3) == (3, 3)
    assert grad_f(7.5, 13) == (13, 8.5)


def test_grad_multi_input_multi_output():
    def func(x, y):
        return x * y, x + y

    ir = make_ir(func)
    grad_f = grad(ir)
    assert grad_f(2, 3) == (3, 1, 2, 1)


def test_hessian():
    def func(x, y):
        return 3 * x * x * x * y + 2 * y

    grad_f = grad(func)
    hess_f = grad(grad_f)
    assert hess_f(2, 3) == (108, 36, 36, 0)


# Execute the tests using pytest
if __name__ == "__main__":
    pytest.main()
