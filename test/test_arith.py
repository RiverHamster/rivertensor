import rivertensor.base as rt
import numpy as np

def test_arith():
    N = 1000423
    a = np.random.randn(N).astype('float32')
    b = np.random.randn(N).astype('float32')
    print(a.shape)
    s_rt = (rt.from_numpy(a) + rt.from_numpy(b)).numpy()
    s = a + b
    assert np.max(np.abs(s_rt - s) < 1e-1)
    a_1_rt = (rt.from_numpy(a) + 1).numpy()
    a_1 = a + 1
    assert np.max(np.abs(a_1_rt - a_1) < 1e-1)

