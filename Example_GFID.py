#GFID Example on the pikachu.jpg image 
import cv2
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

a = cv2.imread('/content/pikachu.jpg')
a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
level = 0.1
_, a_bw = cv2.threshold(a_gray, level*255, 255, cv2.THRESH_BINARY)
Icomp = cv2.bitwise_not(a_bw)
contours, _ = cv2.findContours(Icomp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the boundary in the BW image
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=1)

#number of points in the shape
nbrp=130

#resampling the contour
X1, Y1, L=Reparametrage_euclidien2(contour[:, 0, 0], contour[:, 0, 1], nbrp)

#convert to complex
F = np.zeros(nbrp,  dtype=np.complex128)
for i in range(nbrp):
    F[i] = X1[i] + 1j * Y1[i]
# Ajouter un élément supplémentaire à la fin du tableau F
F = np.append(F, X1[0] + 1j * Y1[0])

IA, theta0, theta1, thetaN=invariant_ghorbel(nbrp, F, 130 , 2, 1, 1, 1)

plt.plot(IA)

