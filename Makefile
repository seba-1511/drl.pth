
.PHONY: all dev

all:
	mpirun -n 4 python mpi_bench.py --algo reinforce --env InvertedPendulum-v1
