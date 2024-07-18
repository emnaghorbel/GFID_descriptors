#GFIDfunction
#nbrp is the number of point on the curve
#F is the complex coordinates of the curve
#Nc is the truncation 
#n0 and n1 are the successive DFT coefficient indexes (n0-n1=1)
#p and q are the power parameters in the GFID expression for the inversion convergence (we fix these parameters to 1 in the following)
#IA is the GFID invariant result. 
import cv2
import numpy as np

def GFID_function(nbrp, F, Nc, n0, n1, p, q):
    # dft
    npc = nbrp // 2
    A = np.zeros(nbrp, dtype=complex)

# Compute A
    for k in range(nbrp):
        for n in range(nbrp):
            A[k] += F[n] * np.exp(-2j * np.pi * k * n / nbrp)

# Centre de gravité
    GA = A[0] / nbrp

# Invariants stables et complets
    An = A[npc:]
    Ap = A[:npc]

# Translation et tronquage
    Ac = np.zeros(nbrp, dtype=complex)
    Ac[:npc] = An
    Ac[npc:] = Ap

    Atr = np.zeros(nbrp, dtype=complex)
    Atr[npc + 1 - Nc:npc + 1 + Nc] = Ac[npc + 1 - Nc:npc + 1 + Nc]

    Anew = Atr[npc + 1 - Nc - 1:npc + 1 - Nc - 1 + 2 * Nc + 1]

    theta0 = np.angle(Anew[n0 + Nc])
    theta1 = np.angle(Anew[n1 + Nc])
    thetaN = np.angle(Anew)

# Calcul des invariants
    Kk = np.zeros(2 * Nc + 1)
    for v in range(2 * Nc + 1):
        Kk[v] = v * (theta1 - theta0) + (n1 * theta0 - n0 * theta1) + (n0 - n1) * thetaN[v]

    E = np.exp(1j * Kk)

    IA = np.zeros(2 * Nc + 1, dtype=complex)
    for v in range(2 * Nc + 1):
        IA[v] = (abs(Anew[v])**(n0 - n1)) * (abs(Anew[n0 + Nc])**p) * (abs(Anew[n1 + Nc])**q) * E[v]


    return IA, theta0, theta1, thetaN
