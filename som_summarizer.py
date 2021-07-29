import pickle
import yaml
import sys

import numpy as np
import xpysom
import h5py
# 1) make and save the deep som
# 2) make and save the wide som
# 3) use the redshift and overlap samples to complete the process


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

        self.depths = norm_config['depths']
        self.mag_min = norm_config.get("mag_min", 22)
        self.mag_max = norm_config.get("mag_max", 28)

        self.dim = som_config['dim']
        self.rate = som_config['rate']
        self.sigma = som_config['sigma']
        self.epochs = som_config['epochs']
        self.use_errors = som_config['use_errors']

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
        self.som.train(data, self.epochs, comm=self.comm)


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

class SOMSummarizer:
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
    def __init__(self, config, comm=None):
        """

        Parameters
        ----------

        config: dict
            Dictionary with complete config info; see example
        comm: mpi4py communicator or None
            Optional; enables parallel training
        """
        self.config = config
        self.comm = comm
        self.rank = 0 if comm is None else self.comm.rank
        self.nproc = 1 if comm is None else self.comm.size

    def log(self, msg, root=True):
        if self.comm is None:
            print(msg)
        else:
            if (not root) or self.rank == 0:
                print(f"Rank {self.rank}: {msg}")

    def run(self):
        """
        Run all the steps in the method, saving results along the way

        Incomplete.
        """
        self.log("\n*** Making deep SOM ***\n")
        deep_som = self.make_som(self.config['deep'])
        self.save_som(deep_som, self.config['output']['deep_som'])

        # We use the full distribution for the actual calculations.
        # This is for plotting
        self.log("\n*** Making deep z distribution ***\n")
        deep_zmaps = self.make_deep_zmaps(deep_som)
        self.save_maps(deep_zmaps, self.config['output']['deep_name'])

        self.log("\n*** Making wide SOM ***\n")
        wide_som = self.make_som(self.config['wide'])
        self.save_som(wide_som, self.config['output']['wide_som'])

        self.log("\n*** Making wide->deep transfer function ***\n")
        transfer = self.make_transfer(deep_som, wide_som)
        if self.rank == 0:
            np.save('transfer.npy', transfer)


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
        config =self.config['overlap']['wide'].copy()

        bands = [f'deep_{b}' for b in deep_som.bands] + [f'wide_{b}' for b in deep_som.bands]

        # errors!
        self.log("TODO: add errors in make_transfer!")

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


    def save_som(self, som, outfile):
        """
        Save a LuptiSom to file. 

        Right now this just pickles it.
        TODO: better serialization

        Parameters
        ----------
        som: LuptiSom

        outfile: str
        """
        if self.rank == 0:
            pickle.dump(som, open(outfile, 'wb'))


    def data_stream(self, config):
        """
        A generator function that yields chunks of loaded data.

        The "format" entry of the configuration determines the
        data source.

        TODO: maybe move to superclass?

        Parameters
        ----------
        config: dict
            Describes the source of the data and what to load

        Yields
        ------
        data: dict
            Dictionaries mapping column -> array
        """
        fmt = config['format']
        if fmt in ["hdf", "hdf5"]:
            return self.hdf5_data_stream(config)
        else:
            raise ValueError(f"Unknown data stream type {fmt}")


    def hdf5_data_stream(self, config):
        """
        HDF5-specific implementation of data_streams.

        TODO: maybe move to superclass?

        Parameters
        ----------
        config: dict
            Describes the source of the data and what to load

        Yields
        ------
        data: dict
            Dictionaries mapping column -> array
        """
        filename = config['file']
        stream = config['stream']
        group = config['group']
        fmt = config.get('band_name', 'mag_{}')
        err_fmt = config.get('error_name', 'mag_err_{}')
        bands = config['bands']
        nmax = config.get('nmax')
        random_order = config.get('random_chunk_order', True)
        order_seed = config.get('order_seed', 1234567)        

        cols = config.get('extra', {}).copy()
        for b in bands:
            cols[b] = fmt.format(b)

        if config['load_errors']:
            for b in bands:
                cols[f'{b}_err'] = err_fmt.format(b)

        with h5py.File(filename) as f:
            g = f[group]
            first_col = list(cols.values())[0]
            sz = g[first_col].size
            if nmax:
                sz = min(nmax, sz)

            if stream:
                max_chunk_size = config.get('max_chunk_size', 100_000)
                # we need a number of chunks that splits the entire
                # data set into chunks which:
                # i) are a multiple of the number of MPI processes
                # ii) are smaller than max_chunk_size
                nchunk = int(np.ceil(sz / max_chunk_size / self.nproc)) * self.nproc
                chunk_size = int(np.ceil(sz / nchunk))
                chunk_starts = np.arange(nchunk) * chunk_size
                chunk_ends = chunk_starts + chunk_size
                chunk_order = np.arange(len(chunk_starts))

                # We (optionally) randomize the order of the chunk.
                # This is because if the input is e.g. ordered by redshift
                # then that could bias the training.
                if random_order:
                    rng = np.random.default_rng(seed=order_seed)
                    rng.shuffle(chunk_order)
                    self.log(f"Will load {nchunk} data chunks in order: {chunk_order}")
                    if self.comm:
                        self.comm.Bcast(chunk_order)

                for i, chunk_index in enumerate(chunk_order):
                    if i % self.nproc != self.rank:
                        continue
                    s = chunk_starts[chunk_index]
                    e = chunk_ends[chunk_index]
                    self.log(f"Reading {filename} data range {s} - {e} (chunk {i+1} / {nchunk})", root=False)
                    data = {name: g[col][s:e] for name, col in cols.items()}
                    yield data

            else:
                block_size = int(np.ceil(sz / self.nproc))
                s = block_size * self.rank
                e = s + block_size
                self.log(f"Reading {filename} its full data range {s} - {e}", root=False)
                data = {name: g[col][s:e] for name, col in cols.items()}

                yield data


    def make_som(self, config):
        """
        Make and train a SOM base on some configuration options.

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




def main():
    config = yaml.safe_load(open("config.yml"))

    try:
        from mpi4py.MPI import COMM_WORLD as world
        comm = None if world.size == 1 else world
    except:
        comm = None

    if comm and comm.rank == 0:
        print("Running under MPI with world size = ", comm.size)
    else:
        print("Running in serial mode")

    ss = SOMSummarizer(config, comm)
    ss.run()

if __name__ == '__main__':
    main()
