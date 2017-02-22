
.PHONY: all dev

all:
	mpirun -n 4 python benchmarks/mpi_bench.py --agent reinforce --env InvertedPendulum-v1
