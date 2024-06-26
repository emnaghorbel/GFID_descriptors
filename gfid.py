#GFID Example on the pikachu.jpg image 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline
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
    
a = cv2.imread('/content/pikachu.jpg')
a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
level = 0.1
_, a_bw = cv2.threshold(a_gray, level*255, 255, cv2.THRESH_BINARY)
Icomp = cv2.bitwise_not(a_bw)
contours, _ = cv2.findContours(Icomp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#plt.imshow(cv2.cvtColor(a_bw, cv2.COLOR_GRAY2RGB))

for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=1)
plt.show()

#number of points in the shape
nbrp=130
#resampling the contour
X1, Y1, L=Reparametrage_euclidien2(contour[:, 0, 0], contour[:, 0, 1], nbrp)
#plt.plot(X1,Y1)

#convert to complex
F = np.zeros(nbrp,  dtype=np.complex128)
for i in range(nbrp):
    F[i] = X1[i] + 1j * Y1[i]
# Ajouter un élément supplémentaire à la fin du tableau F
F = np.append(F, X1[0] + 1j * Y1[0])

#plt.plot(np.real(F), np.imag(F))


IA, theta0, theta1, thetaN=invariant_ghorbel(nbrp, F, 130 , 2, 1, 1, 1)

plt.plot(IA)

