import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.interpolate import CubicSpline
from GFID_function import GFID_function
from Reparametrage_euclidien2 import Reparametrage_euclidien2

# Load and preprocess the image and extract the contour
a = cv2.imread('pikachu.jpg')
a_gray = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
level = 0.1
_, a_bw = cv2.threshold(a_gray, int(level * 255), 255, cv2.THRESH_BINARY)
Icomp = cv2.bitwise_not(a_bw)
contours, _ = cv2.findContours(Icomp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Extract the boundary in the BW image
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    plt.plot(contour[:, 0, 0], contour[:, 0, 1], 'r', linewidth=1)

# Number of points in the shape after resampling
nbrp = 130

# Resampling the contour
X1, Y1, L = Reparametrage_euclidien2(contour[:, 0, 0], contour[:, 0, 1], nbrp)

# Convert to complex
F = np.array([X1[i] + 1j * Y1[i] for i in range(nbrp)], dtype=np.complex128)
#close the curve to obtain a contour (periodicity)
F = np.append(F, X1[0] + 1j * Y1[0])

# Parameters
Nc = 25
n0 = 2
n1 = 1
p = 1
q = 1

npc = nbrp // 2
A = np.zeros(nbrp, dtype=complex)

# Compute A (complex Discrete Fourier Transform)
for k in range(nbrp):
    for n in range(nbrp):
        A[k] += F[n] * np.exp(-2j * np.pi * k * n / nbrp)

# centroide
GA = A[0] / nbrp

# The calculation of the GFID 
An = A[npc:]
Ap = A[:npc]

# Translation et troncation
Ac = np.zeros(nbrp, dtype=complex)
Ac[:npc] = An
Ac[npc:] = Ap

Atr = np.zeros(nbrp, dtype=complex)
Atr[npc + 1 - Nc:npc + 1 + Nc] = Ac[npc + 1 - Nc:npc + 1 + Nc]

Anew = Atr[npc + 1 - Nc - 1:npc + 1 - Nc - 1 + 2 * Nc + 1]

theta0 = np.angle(Anew[n0 + Nc])
theta1 = np.angle(Anew[n1 + Nc])
thetaN = np.angle(Anew)

# GFID Invariants
#!st the power part
Kk = np.zeros(2 * Nc + 1)
for v in range(2 * Nc + 1):
    Kk[v] = v * (theta1 - theta0) + (n1 * theta0 - n0 * theta1) + (n0 - n1) * thetaN[v]

E = np.exp(1j * Kk)
#the final GFID descriptor
IA = np.zeros(2 * Nc + 1, dtype=complex)
for v in range(2 * Nc + 1):
    IA[v] = (abs(Anew[v])**(n0 - n1)) * (abs(Anew[n0 + Nc])**p) * (abs(Anew[n1 + Nc])**q) * E[v]


#THE GFID INVERSION
D = ((n0 - n1 + p) * (n0 - n1 + q)) - (p * q)
alpha_0 = (n0 - n1 + q) / D
beta_0 = -q / D
alpha_1 = (n0 - n1 + p) / D
beta_1 = -p / D
P2 = 1 / (n0 - n1)
P0 = ((-p) * P2) * alpha_0 + beta_1 * (-q * P2)
P1 = ((-q) * P2) * alpha_1 + beta_0 * (-p * P2)

# Simplification du calcul de l'argument
THETA_0 = (theta0 - theta1) / (n0 - n1)
THETA_1 = (n0 * theta0 - n1 * theta1) / (n0 - n1)
EXPP = np.exp(1j * (np.arange(2 * Nc + 1) * THETA_0 + THETA_1) / (n0 - n1))

Fterm = IA**P2
Sterm = (IA[n0 + Nc]**P0) * (IA[n1 + Nc]**P1)

Fn = Fterm * Sterm * EXPP

# Retour
LA = np.zeros(nbrp, dtype=complex)
LA[:Nc + 1] = Fn[Nc:Nc + 1 + Nc]
LA[nbrp - Nc:] = Fn[:Nc]

# IDFT
G = np.zeros(nbrp + 1, dtype=complex)
for n in range(nbrp):
    for k in range(nbrp):
        G[n] += LA[k] * np.exp(2j * np.pi * n * k / nbrp)
    G[n] /= nbrp
G[nbrp] = G[0]

# Plot
plt.figure()
plt.plot(np.real(G), np.imag(G), 'b')
plt.show()
