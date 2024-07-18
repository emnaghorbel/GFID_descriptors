#reconstruct GFID is the inversion fuction of the GFID descriptors where G is the reconstructed shape with some truncation


import numpy as np

def reconstruct_GFID(nbrp, n0, n1, IA, Nc, p, q, theta0, theta1):
    D = ((n0 - n1 + p) * (n0 - n1 + q)) - (p * q)
    alpha_0 = (n0 - n1 + q) / D
    beta_0 = -q / D
    alpha_1 = (n0 - n1 + p) / D
    beta_1 = -p / D
    P2 = 1 / (n0 - n1)
    P0 = ((-p) * P2) * alpha_0 + beta_1 * (-q * P2)
    P1 = ((-q) * P2) * alpha_1 + beta_0 * (-p * P2)
    
    THETA_0 = (theta0 - theta1) / (n0 - n1)
    THETA_1 = (n0 * theta0 - n1 * theta1) / (n0 - n1)
    
    EXPP = np.zeros(2 * Nc + 1, dtype=complex)
    for i in range(2 * Nc + 1):
        EXPP[i] = np.exp(1j * (i * THETA_0 + THETA_1) / (n0 - n1))
        
    Fterm = np.zeros(2 * Nc + 1, dtype=complex)
    for i in range(2 * Nc + 1):
        Fterm[i] = IA[i]**P2
        
    Sterm = (IA[n0 + Nc]**P0) * (IA[n1 + Nc]**P1)
    
    Fn = np.zeros(2 * Nc + 1, dtype=complex)
    for i in range(2 * Nc + 1):
        Fn[i] = Fterm[i] * Sterm * EXPP[i]
        
    LA = np.zeros(nbrp, dtype=complex)
    for i in range(Nc + 1):
        LA[i] = Fn[Nc + i]
        LA[nbrp - Nc + i - 1] = Fn[i]
        
    G = np.zeros(nbrp + 1, dtype=complex)
    for n in range(nbrp):
        for k in range(nbrp):
            G[n] += LA[k] * np.exp(2j * np.pi * n * k / nbrp)
        G[n] /= nbrp
    G[nbrp] = G[0]
    
    return G
