from .summarizer import Summarizer
from .som import LuptiSom
import numpy as np

class SomSummarizer(Summarizer):

    def __init__(self, config, comm=None):
        super().__init__(config, comm=comm)
        self.rng = np.random.default_rng()


    def randomize_data(self, data):
        randomize_selection = self.config['somz']['randomize_selection']
        randomize_noise = self.config['somz']['randomize_noise']

        if not (randomize_selection or randomize_noise):
            return data

        n = list(data.values())[0].size
        # take a random subset of objects, of the same size as the full sample
        # but potentially using some of them twice.
        if randomize_selection:
            idx = np.arange(n)
            sel = np.random.choice(idx, n)
            data_copy = {name: col[sel] for name, col in data.items()}
        else:
            data_copy = {name: col.copy() for name, col in data.items()}


        # randomly perturb those objects.
        # This is in the somz paper but doesn't actually make sense
        # to me - adding noise again doesn't give you another realization
        # from the same sample, because you've then convolved with the
        # noise distribution twice. We could consider deconvolving
        # with e.g. the Bovy et al extreme deconvolution approach
        # and then generating a new realization, though that would of
        # course bring new problems.

        if randomize_noise:
            config = self.config['redshift']['input']
            bands = config['bands']
            band_name = config['band_name']  # Wyld Stallyns
            error_name = config['error_name']

            for b in bands:
                c = data_copy[band_name.format(b)]
                e = data_copy[error_name.format(b)]
                r = self.rng.randn(size=n)
                # add random noise
                c += e * r

        return data_copy



    def run(self):


        # number of random perturbations of the data
        n_r = self.config['somz']['n_r']
        # number of random selections of galaxies in training set
        # (later: also do drop out of colours here)
        n_m = self.config['somz']['n_m']

        # Make a SOM from the spectroscopic data
        self.log("\n*** Making spectroscopic SOM ***\n")

        config = self.config['redshift']

        config['input']['stream'] = False
        data = next(self.data_stream(config['input']))
        n = list(data.values())[0].size

        # split into training and testing

        idx = np.arange(n)
        self.rng.shuffle(idx)
        training = {name: col[idx] for name, col in data.items()}
        # training = {name: col[:n//2] for name, col in data}
        # testing = {name: col[n//2:] for name, col in data}


        # we will mutate the data a little according to the above

        som_config = config['som']
        data_config = config['input']
        norm_config = self.config['norm']

        
        soms = []
        nsom = n_r * n_m
        dim = som_config['dim']
        zsum = np.zeros((nsom, dim, dim))
        zsum2 = np.zeros((nsom, dim, dim))
        count = np.zeros((nsom, dim, dim))
        i = 0
        for r in range(n_r):
            for m in range(n_m):
                print(f"Making SOM {(r+1)*(m+1)}")
                # Make one SOM with randomized data
                som = LuptiSom(som_config, data_config, norm_config, comm=self.comm)

                # make a random data variation and train on it
                data_rand = self.randomize_data(training)
                som.train(data_rand)
                soms.append(som)

                w = som.winner(data_rand)

                zsumi = zsum[i]
                zsum2i = zsum2[i]
                counti = count[i]
                # numba this
                for wi, z in zip(w, training["redshift"]):
                    zsumi[wi] += z
                    zsum2i[wi] += z**2
                    counti[wi] += 1

                i += 1

        # Redshift distribution for each cell for each som
        zmean = zsum / count
        zsigma2 = zsum2 / count - zmean**2



        # make the distribution for this data set

        # now take the wide data and run it through the SOM
        # and get the distribution
        z = np.arange(0.0, 4.0, 0.01)
        nz = np.zeros_like(z)
        for data in self.data_stream(self.config['wide']['input']):
            for i, som in enumerate(soms):
                w = som.winner(data)
                # mean and stdev for each object
                zmeani = zmean[i]
                zsigma2i = zsigma2[i]
                # TODO: vectorize

                for wi in w:
                    sigma2 = zsigma2i[wi]
                    if not sigma2 > 0.0001:
                        continue
                    mu = zmeani[wi]
                    j = int(mu / 0.01)
                    nz[j] += 1

                    # x = np.exp(-0.5 * (z - mu)**2 / sigma2)
                    # # normalize, even in the case where we are cut off
                    # x /= x.sum()
                    # nz += x
                    # if not np.isfinite(nz).all():
                    #     import pdb
                    #     pdb.set_trace()

        nz = nz / nz.sum()
        nz /= 0.01
        import h5py
        with h5py.File('./wide_catalog.hdf5') as f:
            ztrue = f['/metacal/redshift_true'][:]
        print(ztrue)
        # now we have n(z)!
        import pylab
        pylab.hist(ztrue, bins=30, density=1)
        pylab.plot(z, nz)
        pylab.show()


        # print(z)
        # print(nz)

        # choices: could 


        #
        # can make suite of SOMs
        # n_r * n_m SOMs where for each n_m a different random
        # selection is made from the galaxies (full count but random with replacement)
        # and for each n_r noise is randomly added to each
