
.PHONY: all dev mpi

ALGO=reinforce
ENV=InvertedPendulum-v1
N_ITER=400
TEST_N_ITER=100
OPT=Adam
LR=0.001

all: mpi

mpi:
	mpirun -n 8 python mpi_bench.py --algo reinforce --env InvertedPendulum-v1 --n_iter 300 --n_test_iter 10 --timesteps_per_batch 5000 --opt Adam --lr 0.001

dev:
	python benchmark.py --algo reinforce --env InvertedPendulum-v1 --n_iter 400 --n_test_iter 10 --timesteps_per_batch 5000 --opt Adam --lr 0.001
