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



def save_maps(maps, name, plot=True):
    """
    Save generated deep z maps to disc and optionally
    plots them as well.
    """
    count, zmean, zsigma, zhist = maps

    # TODO: save these in a more structured way.
    np.save(f'{name}_zcount.npy', count)
    np.save(f'{name}_zmean.npy', zmean)
    np.save(f'{name}_zsigma.npy', zsigma)
    np.save(f'{name}_zhist.npy', zhist)

    if not plot:
        return

    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    fig, axes = plt.subplots(3, 1, figsize=(4, 10))
    im = axes[0].imshow(count, norm=LogNorm(vmin=1, vmax=count.max()))
    plt.colorbar(im, ax=axes[0])
    axes[0].set_title("count")

    im = axes[1].imshow(zmean)
    plt.colorbar(im, ax=axes[1])
    axes[1].set_title("mean")

    img = axes[2].imshow(zsigma)
    plt.colorbar(im, ax=axes[2])
    axes[2].set_title("std dev")

    fig.savefig(f"{name}.png")



