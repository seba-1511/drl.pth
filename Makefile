
ALGO=ppo
ALGO=reinforce
N_STEPS=100000000
TEST_N_STEPS=100
NUM_WORKERS=2
DROPOUT=0.0

ENV=CartPole-v0
#ENV=InvertedPendulum-v1
#MODEL=fc
MODEL=lstm

ifeq ($(ENV),CartPole-v0)
ifeq ($(MODEL),fc)
LAYER_SIZE=128
OPT=Adam
LR=0.01
endif
ifeq ($(MODEL),lstm)
LAYER_SIZE=16
LR=0.01
OPT=SGD
endif
endif

ifeq ($(ENV),InvertedPendulum-v1)
ifeq ($(MODEL),fc)
LAYER_SIZE=128
LR=0.0001
OPT=Adam
endif
ifeq ($(MODEL),lstm)
LAYER_SIZE=16
LR=0.001
OPT=SGD
endif
endif


.PHONY: all dev 

all: dev

async:
	python async_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --model $(MODEL) --update_frequency 00 --max_path_length 5000

sync:
	python sync_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --model $(MODEL) --update_frequency 00 --max_path_length 5000

dev:
	python benchmark.py --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --model $(MODEL) --dropout $(DROPOUT) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --update_frequency 00 --max_path_length 5000

test:
	for algo in reinforce acreinforce a3c; do \
		for env in CartPole-v0 InvertedPendulum-v1; do \
			for model in fc lstm; do \
				for dropout in 0.0 0.95; do \
					python benchmark.py --algo $$algo --env $$env --n_steps 3 --model $$model --dropout $$dropout --n_test_iter 1 --opt Adam --lr 0.001 --update_frequency 000 --max_path_length 1000; \
				done; \
			done; \
		done; \
	done;
