import numpy as np
import xpysom
import h5py
import pickle
import yaml
import sys
# 1) make and save the deep som
# 2) make and save the wide som
# 3) use the redshift and overlap samples to complete the process


class LupticolorSom:

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

        self.nband = len(self.bands)
        self.nfeat = self.nband - 1 if self.use_band is None else self.nband

        self.comm = comm

        # Make our SOM
        self.som = xpysom.XPySom(self.dim, self.dim, self.nfeat, sigma=self.sigma, learning_rate=self.rate, topology='toroidal')


    def train(self, indata):
        names, data = self.make_normalized_color_data(indata)
        self.som.train(data, self.epochs, comm=self.comm)


    def make_normalized_color_data(self, indata):
        # no degenerate colours
        # i-band + colors
        n = list(indata.values())[0].shape[0]

        outdata = np.empty((n, self.nfeat))
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

        if self.use_band is not None:
            # The Lupticolors are already in roughly the right range 0 -- 1.
            # Normalize the Luptitude to the same range
            depth = self.depths[self.use_band]
            m = luptitude(indata[self.use_band], depth)
            outdata[:, self.nband - 1] = (m - self.mag_min) / (self.mag_max - self.mag_min)
            names.append(self.use_band)

        return names, outdata

    def winner(self, indata):
        _, data = self.make_normalized_color_data(indata)
        return self.som.winner(data)

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Replace the unpicklable mpi communicator with None
        state['comm'] = None
        return state



def luptitude(mag, depth, m0=22.5):
    """
    Convert magnitudes to Luptitudes
    """
    a = 2.5 * np.log10(np.e)
    sigmax = 0.1 * np.power(10, (m0 - depth) / 2.5)
    b = 1.042 * sigmax
    mu0 = m0 - 2.5 * np.log10(b)
    flux = np.power(10, (m0 - mag) / 2.5)
    return mu0 - a * np.arcsinh(0.5 * flux / b)


class SOMSummarizer:
    def __init__(self, config, comm=None):
        self.config = config
        self.comm = comm
        self.rank = 0 if comm is None else self.comm.rank
        self.nproc = 1 if comm is None else self.comm.size

    def run(self):
        deep_som = self.make_som(self.config['deep'])
        self.save_som(deep_som, self.config['output']['deep_som'])

        # Could save this out here also
        deep_zmaps = self.make_deep_zmaps(deep_som)
        self.save_maps(deep_zmaps, self.config['output']['deep_name'])


    def save_som(self, som, outfile):
        if self.rank == 0:
            pickle.dump(som, open(outfile, 'wb'))


    def data_stream(self, config, extra=None):
        filename = config['file']
        stream = config['stream']
        group = config['group']
        fmt = config.get('band_name', 'mag_{}')
        bands = config['bands']
        nmax = config.get('nmax')
        random_order = config.get('random_chunk_order', True)
        order_seed = config.get('order_seed', 1234567)
        extra = extra or {}

        cols = [fmt.format(b) for b in bands]


        with h5py.File(filename) as f:
            g = f[group]
            sz = g[cols[0]].size
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
                    if self.rank == 0:
                        print(f"Will load {nchunk} data chunks in order: {chunk_order}")
                    if self.comm:
                        self.comm.Bcast(chunk_order)

                for i, chunk_index in enumerate(chunk_order):
                    if i % self.nproc != self.rank:
                        continue
                    s = chunk_starts[chunk_index]
                    e = chunk_ends[chunk_index]
                    print(f"Rank {self.rank} reading {filename} data range {s} - {e} (chunk {i+1} / {nchunk})")
                    data = {b: g[col][s:e] for b,col in zip(bands,cols)}
                    for name, col in extra.items():
                        data[name] = g[col][s:e]
                    yield data

            else:
                block_size = int(np.ceil(sz / self.nproc))
                s = block_size * self.rank
                e = s + block_size
                print(f"Rank {self.rank} reading {filename} its full data range {s} - {e}")
                data = {b: g[col][s:e] for b,col in zip(bands,cols)}
                for name, col in extra.items():
                    data[name] = g[col][s:e]

                yield data


    def make_som(self, config):
        som_config = config['som']
        data_config = config['input']
        norm_config = self.config['norm']

        som = LupticolorSom(som_config, data_config, norm_config, comm=self.comm)

        # TODO figure out epoch / iter thing to get training right
        data_stream = self.data_stream(data_config)

        for data in data_stream:
            som.train(data)

        return som

    def make_deep_zmaps(self, som):

        zsum = np.zeros((som.dim, som.dim))
        zsum2 = np.zeros((som.dim, som.dim))
        count = np.zeros((som.dim, som.dim))

        config = self.config["spectroscopic"]
        randomize = config.get('randomize', 0)
        stream = self.data_stream(config, extra={"redshift": config["z"]})


        for data in stream:
            w = som.winner(data)
            for wi, z in zip(w, data["redshift"]):
                zsum[wi] += z
                zsum2[wi] += z**2
                count[wi] += 1
        zmean = zsum / count
        zsigma = np.sqrt(zsum2 / count - zmean**2) 


        return count, zmean, zsigma

    def save_maps(self, maps, name):
        if self.rank != 0:
            return

        count, zmean, zsigma = maps

        pickle.dump(count, open(f'{name}_zcount.pkl', 'wb'))
        pickle.dump(zmean, open(f'{name}_zmean.pkl', 'wb'))
        pickle.dump(zsigma, open(f'{name}_zsigma.pkl', 'wb'))

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
    from mpi4py.MPI import COMM_WORLD as world
    comm = None if world.size == 1 else world

    ss = SOMSummarizer(config, comm)
    ss.run()

if __name__ == '__main__':
    main()
