
ALGO=a3c
#ALGO=acreinforce
ENV=CartPole-v0
#ENV=InvertedPendulum-v1
N_ITER=100
TEST_N_ITER=100
OPT=RMSprop
LR=0.001
NUM_WORKERS=8
POLICY=fc
#POLICY=lstm
DROPOUT=0.0

.PHONY: all dev mpi

all: dev

async:
	python async_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --n_test_iter 100 --opt $(OPT) --lr $(LR)

dev:
	python benchmark.py --algo $(ALGO) --env $(ENV) --n_iter $(N_ITER) --policy $(POLICY) --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr $(LR) --update_frequency 100 --max_path_length 1000 

atari:
	#python async_bench.py --n_proc 8 --algo a3c --env PongDeterministic-v4 --n_iter $(N_ITER) --policy atari --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr 0.0001 --update_frequency 100 --max_path_length 10000 --render True
	python benchmark.py --algo a3c --env PongDeterministic-v4 --n_iter $(N_ITER) --policy atari --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr 0.0001 --update_frequency 100 --max_path_length 10000 --render True

test:
	for algo in reinforce acreinforce a3c; do \
		for env in CartPole-v0 InvertedPendulum-v1; do \
			for policy in fc lstm; do \
				for dropout in 0.0 0.95; do \
					python benchmark.py --algo $$algo --env $$env --n_iter 3 --policy $$policy --dropout $$dropout --n_test_iter 1 --opt Adam --lr 0.001 --update_frequency 000 --max_path_length 1000; \
				done; \
			done; \
		done; \
	done;
