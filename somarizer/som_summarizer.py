import numpy as np
from .som import LuptiSom
from .summarizer import Summarizer

class BaseSomSummarizer(Summarizer):

    def make_som(self, config):
        """
        Make and train a SOM base on some configuration options.

        TODO: Might want to go in an intermediate subclass between this and the concrete som classes?

        Parameters
        ----------
        config: dict
            configuration dictionary, should contain "som", "input", and "norm"
            sub-dicts; see LuptiSom docstring.

        Returns
        -------
        som: LuptiSom
            Trained SOM
        """
        som_config = config['som']
        data_config = config['input']
        norm_config = self.config['norm']

        som = LuptiSom(som_config, data_config, norm_config, comm=self.comm)

        for data in self.data_stream(data_config):
            som.train(data)

        return som
    def make_deep_zmaps(self, som):
        """
        Assign data with secure redshifts (e.g. spectroscopic or narrow-band)
        to a (deep) SOM.

        Uses the "redshift" section of the configuration.

        Parameters
        ----------
        som: LuptiSom
            Pre-trained SOM

        Returns
        -------
        count: array
            2D array of number of objects per cell

        zmean: array
            2D array of mean z per cell

        zsigma: array
            2D array of standard deviation per cell

        zhist: array
            3D array of n(z) histogram per cell
        """

        zsum = np.zeros((som.dim, som.dim))
        zsum2 = np.zeros((som.dim, som.dim))
        count = np.zeros((som.dim, som.dim))

        config = self.config["redshift"]
        data_config = config["data"]
        stream = self.data_stream(data_config)

        # Parameters of the histogram per cell
        dz = config["dz"]
        zmax = config["zmax"]
        nz = int(np.ceil(zmax / dz))
        zhist = np.zeros((som.dim, som.dim, nz))

        # if needed we xoult make this way faster
        # could probably just do it with numba
        for data in stream:
            w = som.winner(data)
            for wi, z in zip(w, data["redshift"]):
                zsum[wi] += z
                zsum2[wi] += z**2
                count[wi] += 1
                ni = int(np.floor(z / dz))
                if ni < nz:
                    zhist[wi][ni] += 1
        zmean = zsum / count
        zsigma = np.sqrt(zsum2 / count - zmean**2) 

        return count, zmean, zsigma, zhist

    def save_maps(self, maps, name, plot=True):
        """
        Save generated deep z maps to disc and optionally
        plots them as well.
        """
        if self.rank != 0:
            return

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



