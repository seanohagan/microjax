from microjax import jit
from microjax import grad
import time


def test_jit_simple():
    def f(x):
        return x + 1.0

    jitted_f = jit(f)
    print(str(jitted_f))
    print(f(1))
    print(jitted_f(1))
    assert f(1) == jitted_f(1)


def test_jit_many_input_one_output():
    def f(x, y, z):
        return 2 * x + 3.0 * y * y + (-1 * z)

    jitted_f = jit(f)
    assert f(1, 2, 3) == jitted_f(1, 2, 3)


def test_jit_many_inputs_many_output():
    def f(x, y):
        return (2 * x * x, x * y)

    jitted_f = jit(f)
    assert f(3, 4) == jitted_f(3, 4)


def test_jit_speed():
    def f(x):
        c = 0
        for _ in range(100):
            c += x
        return c

    jitted_f = jit(f)

    times_normal_f = 0
    times_jitted_f = 0

    for _ in range(100):
        start = time.perf_counter()
        f(4)
        times_normal_f += time.perf_counter() - start
        start = time.perf_counter()
        jitted_f(4)
        times_jitted_f += time.perf_counter() - start

    print(times_normal_f / times_jitted_f)
    assert times_jitted_f <= times_normal_f


def test_jit_speed_2():
    def f(x, y, z):
        return 2 * x + 3.0 * y * y + (-1 * z * z * z)

    hess_f = grad(grad(f))
    jitted_hess_f = jit(grad(grad(f)))

    times_normal_f = 0
    times_jitted_f = 0

    for _ in range(100):
        start = time.perf_counter()
        hess_f(4, 5, 6)
        times_normal_f += time.perf_counter() - start
        start = time.perf_counter()
        jitted_hess_f(4, 5, 6)
        times_jitted_f += time.perf_counter() - start

    print(times_normal_f / times_jitted_f)
    assert times_jitted_f <= times_normal_f
