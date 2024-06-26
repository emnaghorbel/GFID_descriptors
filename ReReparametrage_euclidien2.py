def Reparametrage_euclidien2(X, Y, N):
    n = len(X)
    t = np.linspace(0, 1, n)  # parametrisation initiale
    x1 = X
    px = CubicSpline(t, x1)
    y1 = Y
    py = CubicSpline(t, y1)
    L, s = AbscisseEuclidien(t, px, py, N)
    np.save('s.npy', s)
    np.save('px.npy', px)
    X1 = px(s)
    Y1 = py(s)
    X1 = X1.T
    Y1 = Y1.T
    return X1, Y1, L
def AbscisseEuclidien(t, p1, p2, N):
    dp1 = p1.derivative()
    dp2 = p2.derivative()
    X1 = dp1(t)
    X2 = dp2(t)
    X = np.sqrt(X1**2 + X2**2)
    pp = CubicSpline(t, X)
    I = pp.antiderivative()
    s = I(t)
    L = np.max(np.abs(s))
    s = s / np.max(np.abs(s))
    s, index = np.unique(s, return_index=True)
    out = np.interp(np.linspace(0, 1, N), s, t[index])
    return L, out
