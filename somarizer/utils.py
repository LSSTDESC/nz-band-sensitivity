import numpy as np


def luptitude(mag, depth, m0=22.5):
    """
    Convert magnitudes to luptitudes

    Parameters
    ----------
    mag: float or array

    depth: float
        estimate of e.g. 10 sigma depth

    m0: float
        optional mag zero point (default 22.5)

    Returns
    -------
    lupt: float or array
    """
    a = 2.5 * np.log10(np.e)
    sigmax = 0.1 * np.power(10, (m0 - depth) / 2.5)
    b = 1.042 * sigmax
    mu0 = m0 - 2.5 * np.log10(b)
    flux = np.power(10, (m0 - mag) / 2.5)
    return mu0 - a * np.arcsinh(0.5 * flux / b)

def luptitude_error(mag, mag_err, depth, m0=22.5):
    """
    Convert magnitude errors to luptitude errors

    Parameters
    ----------

    mag: float or array

    mag_err: float or array

    depth: float
        estimate of e.g. 10 sigma depth

    m0: float
        optional mag zero point (default 22.5)

    Returns
    -------
    err: float or array
    """
    a = 2.5 * np.log10(np.e)
    flux = np.power(10, (m0 - mag) / 2.5)
    sigmax = 0.1 * np.power(10, (m0 - depth) / 2.5)
    b = 1.042 * sigmax
    dLdm = np.log(10) * a * mag / np.sqrt(flux**2 + 4 * b**2) / 2.5
    dL = abs(dLdm * mag_err)
    return dL
