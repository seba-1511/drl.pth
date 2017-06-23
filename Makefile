
ALGO=reinforce
ENV=CartPole-v0
ENV=InvertedPendulum-v1
N_ITER=1000
TEST_N_ITER=100
OPT=RMSprop
LR=0.001
NUM_WORKERS=8
POLICY=fc
#POLICY=lstm
DROPOUT=0.0

.PHONY: all dev mpi

all: dev

mpi:
	mpirun -n $(NUM_WORKERS) python mpi_bench.py --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --n_test_iter 100 --opt $(OPT) --lr $(LR)

async:
	python async_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --n_test_iter 100 --opt $(OPT) --lr $(LR)

dev:
	python benchmark.py --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --policy $(POLICY) --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr $(LR) --update_frequency 000 --max_path_length 1000 --render 1
