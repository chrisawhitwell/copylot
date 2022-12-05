def _specPortion_check(R2,minR2,kn,kL,bn):
    tcont = 0
    if R2<minR2:
        tcont=1
    if kn[0]>kn[1]:
        tcont=1
    if kL<3:
        tcont=1
    if (bn[2]<bn[0]) or (bn[2]>bn[1]):
        tcont=1
    return tcont