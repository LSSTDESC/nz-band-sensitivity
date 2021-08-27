import numpy as np
from .som import LuptiSom
from .summarizer import Summarizer
from .utils import save_maps

class PhenoSummarizer(Summarizer):
    """
    Summarization step that estimates an n(z) using three samples,
    wide, deep, and redshift (spectroscopy or narrow-band), with some overlap between
    the wide and deep samples.

    Generates two SOMs, for the deep and wide samples, determines the redshift
    distribution of the deep sample using the redshift sample, and then uses the
    overlapping data between deep and wide to compute the transfer function between
    two SOMs.

    The run method is the main entry point.

    """


    def run(self):
        """
        Run all the steps in the method, saving results along the way

        Incomplete.
        """
        self.log("\n*** Making deep SOM ***\n")
        deep_som = self.make_som(self.config['deep'])

        if self.rank == 0:
            deep_som.save(self.config['output']['deep_som'])

        # We use the full distribution for the actual calculations.
        # This is for plotting
        self.log("\n*** Making deep z distribution ***\n")
        deep_zmaps = self.make_deep_zmaps(deep_som)

        if self.rank == 0:
            save_maps(deep_zmaps, self.config['output']['deep_name'])

        self.log("\n*** Making wide SOM ***\n")
        wide_som = self.make_som(self.config['wide'])

        if self.rank == 0:
            wide_som.save(self.config['output']['wide_som'])

        # Not yet complete:
        # self.log("\n*** Making wide->deep transfer function ***\n")
        # transfer = self.make_transfer(deep_som, wide_som)
        # if self.rank == 0:
        #     np.save('transfer.npy', transfer)


    def make_transfer(self, deep_som, wide_som):
        """
        Compute the transfer function between the wide and deep SOMS.

        Reads overlap data according to the config information.

        Not tested yet because I need to generate some mock overlap data
        with both wide and deep values.

        Parameters
        ----------

        deep_som: LuptiSom
            SOM trained on the deep data

        wide_som: LuptiSom
            SOM trained on the shallow data

        Returns
        -------
        transfer: array
            4D array mapping deep to wide cells. For each i, j, k, l
            The array gives the fraction of objects in wide cell k, l
            that land in deep cell i, j.

        """
        config = self.config['overlap']['wide'].copy()

        bands = [f'deep_{b}' for b in deep_som.bands] + [f'wide_{b}' for b in deep_som.bands]

        config['bands'] = bands

        # It's never useful to randomize this data
        config['randomize'] = False
        
        joint_count = np.zeros((deep_som.dim, deep_som.dim, wide_som.dim, wide_som.dim))
        deep_count = np.zeros((deep_som.dim, deep_som.dim))

        for data in self.data_stream(config):
            # get the wide cells
            wide = {b: data[f'wide_{b}'] for b in wide_som.bands}
            wide_cells = wide_som.winner()
            # get the deep cells
            deep = {b: data[f'deep_{b}'] for b in deep_som.bands}
            deep_cells = deep_som.winner()

            # build up the transfer function

            for c1, c2 in zip(wide_cells, deep_cells):
                deep_count[c2] += 1
                joint_count[c2][c1] += 1

        # Sum the counts from all of the processors
        if self.comm:
            self.comm.Reduce(deep_count)
            self.comm.Reduce(joint_count)
        
        # This is the transfer function
        joint_count /= deep_count[:, :, np.newaxis, np.newaxis]
        
        return joint_count


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
        data_config = config["input"]
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
