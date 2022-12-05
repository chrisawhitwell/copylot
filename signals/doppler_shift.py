def doppler_shift(c,vt,vs,fo):
    """
    Correct for Doppler shift

    Parameters
    -------------
    c: float
        Propogation speed of waves in the medium
    vt: float
        Speed of reciever relative to medium
    vs: float
        Speed of source relative to medium
    fo: float
        Observed frequency

    Returns
    -------------
    f: float
        Emitted frequency
    """

    ratio =  (c+vt)/(c+vs)
    f = fo/ratio
    return f