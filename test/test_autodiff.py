import rivertensor as rt
from rivertensor import base
import numpy as np
eps = 1e-4

# test AD with scalar function of single argument
def check_grad(f, x):
    x.require_grad = True
    print(x)
    y = f(x)
    print(y)
    y.backward()
    delta = base.randn(x.shape) * eps
    x1 = rt.tensor(x.value + delta)
    y1 = f(x1)
    q = (y1 - y).numpy() / eps
    d = base.inner(x.grad, delta) / eps
    print(x.grad.numpy(), q, d)
    assert np.abs(q - d) < 0.1

def test_1():
    x = rt.randn([4, 5])
    x.requires_grad = True
    w = rt.randn([5, 3])
    k = rt.randn([4, 3])
    check_grad(
        lambda x: rt.inner(x @ w, k),
        x
    )

def test_1():
    x = rt.randn([4, 5])
    x.requires_grad = True
    w = rt.randn([5, 3])
    k = rt.randn([4, 3])
    check_grad(
        lambda x: rt.inner(rt.sigmoid(x @ w), k),
        x
    )

def test_2():
    x = rt.randn([4, 5])
    x.requires_grad = True
    w = rt.randn([5, 3])
    k = rt.randn([4, 3])
    check_grad(
        lambda x: rt.inner(rt.relu(x @ w), k),
        x
    )

def test_3():
    x = rt.randn([4, 5])
    x.requires_grad = True
    w = rt.randn([5, 3])
    labels = np.random.randint(0, 3, 4).tolist()
    check_grad(
        lambda x: rt.cross_entropy(x @ w, labels),
        x
    )
