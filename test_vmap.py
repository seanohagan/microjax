from microjax import vmap


def test_vmap_single():
    def f(x):
        return x + 10

    vf = vmap(f)

    input = [1, 2, 3, 4]
    assert vf(input) == [f(x) for x in input]


def test_vmap_multi():
    def g(x, y, z):
        return (x + 2 * y, z * z)

    vg = vmap(g)
    input = [(1, 2, 3), (4, 5, 6)]
    assert vg(input) == [g(*x) for x in input]
