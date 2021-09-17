import pickle
import numpy as np
import xpysom
from .utils import luptitude, luptitude_error

class LuptiSom:
    """
    Make and train a self-organizing map.

    This wraps a modified version of the xpysom library (TODO: move this to DESC),
    and converts the loaded magnitudes into luptitudes and normalizes these before
    applying the SOM to them.

    Can run in parallel with MPI by supplying an mpi4py communicator or on a GPU
    automatically if the cupy library is installed and one is available

    Parameters
    ----------

    som_config: dict
        Dictionary of parameters describing the data to use. Contains:
        bands: list or string of names to use
        use_band: optional string of band to include directly in the SOM

    data_config: dict
        Dictionary of parameters describing the SOM. Contains:
        dim: the size of the SOM (it is square)
        rate: the initial learning rate
        sigma: the initial kernel size
        epochs: training steps per data point
        use_error: (not working yet) include errors in the distance calculations

    norm_config: dict
        Dictionary of parameters decsribing how to normalize data. COntains:
        depths: dict of bands -> approx 10 sigma depths for luptitudes
        mag_min: min value for mag/lupt normalization. Does not need to be exact.
        mag_max: max value for mag/lupt normalization. Does not need to be exact.

    comm: MPI communicator or None
        Optional communicator to enable MPI training

    Attributes
    ----------
    som: xpysom.XPySom
        The xpysom self-organizing map
    bands: str or list
        The bands used in the fit
    nfeat: int
        Number of features used in fit
    dim: int
        The dimension of the SOM
    """

    def __init__(self, som_config, data_config, norm_config, comm=None):
        self.bands = data_config['bands']
        self.use_band = data_config.get('use_band', None)
        print("Making SOM with bands: ", "".join(self.bands))

        self.depths = norm_config['depths']
        self.mag_min = norm_config.get("mag_min", 22)
        self.mag_max = norm_config.get("mag_max", 28)

        self.dim = som_config['dim']
        self.rate = som_config['rate']
        self.sigma = som_config['sigma']
        self.epochs = som_config['epochs']
        self.use_errors = som_config['use_errors']
        self.verbose = som_config.get("verbose", "False")

        if self.use_errors:
            raise NotImplementedError("use_errors not yet supported")

        self.nband = len(self.bands)
        self.nfeat = self.nband - 1 if self.use_band is None else self.nband

        self.comm = comm

        # Make our SOM
        self.som = xpysom.XPySom(self.dim, self.dim, self.nfeat, sigma=self.sigma, learning_rate=self.rate, topology='toroidal')


    def train(self, indata):
        """
        Train the SOM with (a slice of) data.

        Can be called multiple times with different data chunks, and called
        in parallel

        Parameters
        ----------
        indata: dict
            dictionary of columns band -> array and optionally errors band_err -> array

        """
        names, data, errors = self.make_normalized_color_data(indata)
        # TODO: add use of errors in the SOM, and fix the luptitude errors
        # TODO: check how the training rate is being changed when this is called
        #       multiple times
        self.som.train(data, self.epochs, comm=self.comm, verbose=self.verbose)


    def make_normalized_color_data(self, indata):
        """
        Convert input magnitudes to LuptiColors and normalize them.

        Optionally, one Luptitude is included as well as the LuptiColors,
        depending on the configuration used on init.

        Parameters
        ----------
        indata: dict
            dictionary of columns band -> array and optionally errors band_err -> array

        Returns
        -------
        names: list
            Names of colors and bands used

        outdata: array 
            ndata x nfeat array ready for training

        errors: array
            ndata x nfeat aray of errors on each training value
        """
        # no degenerate colours
        # i-band + colors
        n = list(indata.values())[0].shape[0]

        outdata = np.empty((n, self.nfeat))
        if self.use_errors:
            errors = np.empty((n, self.nfeat))
        else:
            errors = None
        names = []
        for i in range(self.nband - 1):
            j = i + 1
            b_i = self.bands[i]
            b_j = self.bands[j]
            depth_i = self.depths[b_i]
            depth_j = self.depths[b_j]
            names.append(f"{b_i}-{b_j}")

            # make the lupticolors - differences of luptitudes
            outdata[:, i] = luptitude(indata[b_i], depth_i) - luptitude(indata[b_j], depth_j)

            if self.use_errors:
                mag_err_i = indata[f'{b_i}_err']
                mag_err_j = indata[f'{b_j}_err']
                err_i = luptitude_error(indata[b_i], mag_err_i, depth_i)
                err_j = luptitude_error(indata[b_j], mag_err_j, depth_j)
                errors[:, i] = np.sqrt(err_i**2 + err_j**2)


        if self.use_band is not None:
            # The Lupticolors are already in roughly the right range 0 -- 1.
            # Normalize the Luptitude to the same range
            depth = self.depths[self.use_band]
            m = luptitude(indata[self.use_band], depth)
            outdata[:, self.nband - 1] = (m - self.mag_min) / (self.mag_max - self.mag_min)
            names.append(self.use_band)

            if self.use_errors:
                mag_err = indata[f'{self.use_band}_err']
                err = luptitude_error(indata[self.use_band], mag_err, depth)
                errors[:, self.nband - 1] = err

        return names, outdata, errors

    def winner(self, indata):
        """
        After training is complete, use the SOM to determine the best-fitting
        cell for a set of objects.

        TODO: modify xpysom to return this is an n x 2 array

        Parameters
        ----------
        indata: dict
            dictionary of columns band -> array and optionally errors band_err -> array

        Returns
        -------
        winners: list
            list of tuples of coordinates 
        """
        _, data, _ = self.make_normalized_color_data(indata)
        return self.som.winner(data)

    def __getstate__(self):
        # This lets us pickle this class
        state = self.__dict__.copy()
        # Replace the unpicklable mpi communicator with None
        state['comm'] = None
        return state

    def save(self, outfile):
        """
        Save self to file. 

        Right now this just pickles it.
        TODO: better serialization

        Parameters
        ----------
        outfile: str
        """
        pickle.dump(self, open(outfile, 'wb'))
