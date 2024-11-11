import numpy as np
import pytensor as pt
import random

random.seed(0)
np.random.seed(0)

def softmax(x):
    x -= np.max(x, axis=-1, keepdims=True)
    x = np.exp(x)
    return x / np.sum(x, axis=-1, keepdims=True)

def CELoss(x, labels):
    x = softmax(x)
    return np.mean(-np.log(x[range(x.shape[0]), labels]))

def T_softmax(n, mu=0, sigma=1):
    x = np.random.randn(n) * sigma + mu
    sm_pt = pt.randn([n])
    pt.softmax(pt.from_numpy(x), sm_pt)
    sm_pt = sm_pt.numpy()
    sm = softmax(x)
    assert np.max(np.abs(sm_pt - sm) < 1e-3)

def T_celoss(n, c, mu=0, sigma=1):
    x = np.random.randn(n, c) * sigma + mu
    labels = np.random.randint(c, size=[n])
    loss_pt = pt.CELoss(pt.from_numpy(x), labels.tolist())
    loss = CELoss(x, labels.tolist())
    assert np.max(np.abs(loss_pt - loss) < 1e-3)

def T_celoss_grad(n, c):
    x = pt.randn([n, c])
    eps = 1e-3
    dx = pt.randn([n, c]) * eps
    labels = np.random.randint(c, size=[n])
    loss = pt.CELoss(x, labels.tolist())
    loss1 = pt.CELoss(x + dx, labels.tolist())
    grad = pt.randn([n, c])
    pt.CELoss_grad(x, labels.tolist(), grad)
    dloss = loss1 - loss
    dloss_pred = pt.inner(dx, grad)
    assert np.abs(dloss - dloss_pred) / np.max([loss, loss1, dloss, 1.0]) / eps < 0.1

def test_softmax():
    for i in range(20):
        n = random.randint(1, 100)
        mu = (random.random() - 0.5) * 10
        sigma = random.random()
        T_softmax(n, mu, sigma)

def test_celoss():
    for i in range(200):
        n = random.randint(1, 10)
        c = random.randint(1, 40)
        mu = (random.random() - 0.5) * 10
        sigma = random.random()
        T_celoss(n, c, mu, sigma)

def test_celoss_grad():
    for i in range(200):
        n = random.randint(1, 10)
        c = random.randint(1, 40)
        T_celoss(n, c)
