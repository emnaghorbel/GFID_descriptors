#GFIDfunction
#nbrp is the number of point of the curve
#F is the complex coordinates of the curve
#Nc is the truncation 
#n0 and n1 are the successive DFT coefficient indexes (n0-n1=1)
#p and q are the power parameters in the GFID expression for the inversion convergence (we fix these parameters to 1 in the following)
#IA is the GFID invariant result. 

def GFID_function(nbrp, F, Nc, n0, n1, p, q):
    # dft
    npc = nbrp // 2
    A = np.zeros(nbrp, dtype=np.complex128)
    for k in range(nbrp):
        for n in range(nbrp):
            A[k] = A[k] + F[n] * np.exp(-2 * 1j * np.pi * (k - 1) * (n - 1) / nbrp)

    # le centre de gravité est
    GA = A[0] / nbrp

    # invariants stables et complets
    # Partie négative des invariants dans le vecteur An
    An = np.zeros(npc, dtype=np.complex128)
    for i in range(npc + 1, nbrp):
        An[i - npc] = A[i]

    # Partie positive des invariants dans le vecteur Ap
    Ap = np.zeros(npc, dtype=np.complex128)
    for i in range(npc):
        Ap[i] = A[i]

    # translation et tronquage
    Ac = np.zeros(nbrp, dtype=np.complex128)
    for i in range(npc):
        Ac[i] = An[i]
    for i in range(npc, nbrp):
        Ac[i] = Ap[i - npc]

    Atr = np.zeros(2 * Nc + 1, dtype=np.complex128)
    # Adjust the loop to ensure valid indices for Atr
    for i, v in enumerate(range(npc + 1 - Nc, npc + 1 + Nc)):
        if 0 <= v < len(Ac):  # Check if v is a valid index for Ac
            Atr[i] = Ac[v]

    Anew = np.zeros(2 * Nc + 1, dtype=np.complex128)
    for v in range(2 * Nc + 1):
        Anew[v] = Atr[npc + 1 - Nc - 1 + v]

    theta0 = np.angle(Anew[n0 + 1 + Nc])
    theta1 = np.angle(Anew[n1 + 1 + Nc])
    thetaN = np.angle(Anew)

    # calcul des invariants
    Kk = np.zeros(2 * Nc + 1, dtype=np.complex128)
    for v in range(2 * Nc + 1):
        Kk[v] = v * (theta1 - theta0) + (n1 * theta0 - n0 * theta1) + (n0 - n1) * thetaN[v]

    E = np.zeros(2 * Nc + 1, dtype=np.complex128)
    for v in range(2 * Nc + 1):
        E[v] = np.exp(1j * Kk[v])


    IA = np.zeros(2 * Nc + 1, dtype=np.complex128)
    for v in range(2 * Nc + 1):
        IA[v] = (np.abs(Anew[v])**(n0 - n1)) * (np.abs(Anew[n0 + 1 + Nc])**p) * (np.abs(Anew[n1 + 1 + Nc])**q) * E[v]

    return IA, theta0, theta1, thetaN
