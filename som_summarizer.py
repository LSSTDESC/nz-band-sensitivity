import pickle
import yaml
import sys

import numpy as np
import xpysom
import h5py
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
        self.use_errors = som_config['use_errors']

        if self.use_errors:
            raise NotImplementedError("use_errors not yet supported")

        self.nband = len(self.bands)
        self.nfeat = self.nband - 1 if self.use_band is None else self.nband

        self.comm = comm

        # Make our SOM
        self.som = xpysom.XPySom(self.dim, self.dim, self.nfeat, sigma=self.sigma, learning_rate=self.rate, topology='toroidal')


    def train(self, indata):
        names, data, errors = self.make_normalized_color_data(indata)
        # TODO: add use of errors in the SOM, and fix the luptitude errors
        self.som.train(data, self.epochs, comm=self.comm)


    def make_normalized_color_data(self, indata):
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

def luptitude_error(mag, mag_err, depth, m0=22.5):
    a = 2.5 * np.log10(np.e)
    flux = np.power(10, (m0 - mag) / 2.5)
    sigmax = 0.1 * np.power(10, (m0 - depth) / 2.5)
    b = 1.042 * sigmax
    dLdm = np.log(10) * a * mag / np.sqrt(flux**2 + 4 * b**2) / 2.5
    dL = abs(dLdm * mag_err)
    return dL

class SOMSummarizer:
    def __init__(self, config, comm=None):
        self.config = config
        self.comm = comm
        self.rank = 0 if comm is None else self.comm.rank
        self.nproc = 1 if comm is None else self.comm.size

    def run(self):
        deep_som = self.make_som(self.config['deep'])
        self.save_som(deep_som, self.config['output']['deep_som'])

        # We use the full distribution for the actual calculations.
        # This is for plotting
        deep_zmaps = self.make_deep_zmaps(deep_som)
        self.save_maps(deep_zmaps, self.config['output']['deep_name'])

        # wide_som = self.make_som(self.config['wide'])
        # self.save_som(wide_som, self.config['output']['wide_som'])

        # self.make_transfer(deep_som, wide_som)


    def make_transfer(self, deep_som, wide_som):

        config =self.config['overlap']['wide'].copy()

        bands = [f'deep_{b}' for b in deep_som.bands] + [f'wide_{b}' for b in deep_som.bands]

        # errors!
        print("add errors in make_transfer")

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
        if self.rank == 0:
            pickle.dump(som, open(outfile, 'wb'))


    def data_stream(self, config, extra=None):
        filename = config['file']
        stream = config['stream']
        group = config['group']
        fmt = config.get('band_name', 'mag_{}')
        err_fmt = config.get('error_name', 'mag_err_{}')
        bands = config['bands']
        nmax = config.get('nmax')
        random_order = config.get('random_chunk_order', True)
        order_seed = config.get('order_seed', 1234567)        
        extra = extra or {}

        cols = [fmt.format(b) for b in bands]
        if config['load_errors']:
            for b in bands:
                extra[f'{b}_err'] = err_fmt.format(b)


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
                else:
                    yield data


    def make_som(self, config):
        som_config = config['som']
        data_config = config['input']
        norm_config = self.config['norm']

        som = LupticolorSom(som_config, data_config, norm_config, comm=self.comm)

        for data in self.data_stream(data_config):
            print(data.keys())
            som.train(data)

        return som

    def make_deep_zmaps(self, som):

        zsum = np.zeros((som.dim, som.dim))
        zsum2 = np.zeros((som.dim, som.dim))
        count = np.zeros((som.dim, som.dim))

        config = self.config["redshift"]
        data_config = config["data"]
        stream = self.data_stream(data_config, extra={"redshift": data_config["z"]})

        dz = config["dz"]
        zmax = config["zmax"]
        nz = int(np.ceil(zmax / dz))
        zhist = np.zeros((som.dim, som.dim, nz))

        # if needed we make this way faster
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
    from mpi4py.MPI import COMM_WORLD as world
    comm = None if world.size == 1 else world

    ss = SOMSummarizer(config, comm)
    ss.run()

if __name__ == '__main__':
    main()
