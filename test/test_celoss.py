import numpy as np
import rivertensor.base as rt
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
    sm_rt = rt.softmax(rt.from_numpy(x))
    sm_rt = sm_rt.numpy()
    sm = softmax(x)
    assert np.max(np.abs(sm_rt - sm) < 1e-3)

def T_celoss(n, c, mu=0, sigma=1):
    x = np.random.randn(n, c) * sigma + mu
    labels = np.random.randint(c, size=[n])
    loss_rt = rt.CELoss(rt.from_numpy(x), labels.tolist())
    loss = CELoss(x, labels.tolist())
    assert np.max(np.abs(loss_rt - loss) < 1e-3)

def T_celoss_grad(n, c):
    x = rt.randn([n, c])
    eps = 1e-3
    dx = rt.randn([n, c]) * eps
    labels = np.random.randint(c, size=[n])
    loss = rt.CELoss(x, labels.tolist())
    loss1 = rt.CELoss(x + dx, labels.tolist())
    grad = rt.randn([n, c])
    rt.CELoss_grad(x, labels.tolist(), grad)
    dloss = loss1 - loss
    dloss_pred = rt.inner(dx, grad)
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
