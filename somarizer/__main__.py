import yaml
import sys
from .summarizer import Summarizer
from . import phenoz, somz

def main(config_filename, mpi=True):
    config = yaml.safe_load(open(config_filename))

    if mpi:
        from mpi4py.MPI import COMM_WORLD as world
        comm = None if world.size == 1 else world
    else:
        comm = None

    if comm and comm.rank == 0:
        print("Running under MPI with world size = ", comm.size)
    else:
        print("Running in serial mode")

    summarizer_config = config['summarizer']
    subclass_name = summarizer_config['class']
    subclass = Summarizer.get_subclass(subclass_name)
    if subclass is None:
        raise ValueError(f"Unknown subclass {subclass_name}")

    summarizer = subclass(config, comm)
    summarizer.run()

if __name__ == '__main__':
    main(sys.argv[1])
