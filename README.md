# nz-band-sensitivity

## Getting ready

Get data from:

    https://portal.nersc.gov/cfs/lsst/zuntz/data/deep_photometry_catalog.hdf5
    https://portal.nersc.gov/cfs/lsst/zuntz/data/spectroscopic_data.hdf5

Install:

    pip install -r requirements.txt

## Running locally

Run:

    python som_summarizer.py

Will run as far as the wide som (needs large data set).

## Running at NERSC

First make sure you have access to the Cori CPU nodes: https://docs-dev.nersc.gov/cgpu/

Create an interactive job with two GPUs, twenty CPUs, and one hour with:

    module load cgpu
    salloc -C gpu -t 60 -c 20 -G 2 -q interactive -A m1727

I have made a shifter image with the dependencies in.  Launch with:

    srun -C gpu -u -n 2 -c 10 --cpu-bind=cores --gpus-per-task=1 --gpu-bind=map_gpu:0,1 shifter --image joezuntz/txpipe-2004 python som_summarizer.py

This assigns one CPU to each MPI process, runs two MPI processes with 10 CPUs on each, under shifter, and with unbuffered output.

## Roadmap

- Make a deep/wide overlap sample
- Implement the calculations to get n(z) from DES Y3 paper

