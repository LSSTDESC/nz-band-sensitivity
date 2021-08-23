import pickle
import numpy as np
from .som import LuptiSom


class Summarizer:
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

    _summarizer_classes = {}
    @classmethod
    def __init_subclass__(cls, **kwargs):
        # Run any parent subclassing that is needed
        super().__init_subclass__(**kwargs)
        # Store this class in a dict
        print(cls.__name__, cls)
        cls._summarizer_classes[cls.__name__] = cls


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




    @classmethod
    def get_subclass(cls, name):
        """
        Return the Summarizer subclass with the given name.

        Returns
        -------
        subclass: class
            The corresponding subclass
        """
        return cls._summarizer_classes.get(name)


    def log(self, msg, root=True):
        if self.comm is None:
            print(msg)
        else:
            if (not root) or self.rank == 0:
                print(f"Rank {self.rank}: {msg}")

    def run(self):
        raise NotImplementedError("run method")



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
        import h5py
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

