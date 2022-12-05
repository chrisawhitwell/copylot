def f2k(Pf,f,U):
    Pk = Pf * U/(2*np.pi)
    k = f*2*np.pi/U
    return Pk,k