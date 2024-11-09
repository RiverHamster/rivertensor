import pytensor as pt
import numpy as np

def test_arith():
    N = 1000423
    a = np.random.randn(N).astype('float32')
    b = np.random.randn(N).astype('float32')
    print(a.shape)
    s_pt = (pt.from_numpy(a) + pt.from_numpy(b)).numpy()
    s = a + b
    assert np.max(np.abs(s_pt - s) < 1e-1)
    a_1_pt = (pt.from_numpy(a) + 1).numpy()
    a_1 = a + 1
    assert np.max(np.abs(a_1_pt - a_1) < 1e-1)

