
ALGO=random
ALGO=trpo
ENV=InvertedPendulum-v1
N_ITER=100000000
TEST_N_ITER=100
OPT=RMSprop
LR=0.001
NUM_WORKERS=8

.PHONY: all dev mpi

all: dev

mpi:
	mpirun -n $(NUM_WORKERS) python mpi_bench.py --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --n_test_iter 100 --opt $(OPT) --lr $(LR)

dev:
	python benchmark.py --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --n_test_iter 100 --opt SGD --lr 0.01 --timesteps_per_batch 15000 --max_path_length 5000

async:
	python async_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --n_test_iter 100 --opt $(OPT) --lr $(LR)
