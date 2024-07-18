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

   
  return G
