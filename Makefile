
# TODO:
# - Cleanup PPO
# - Fix PPO (learning is not as efficient as A2C for now. Suspect: GAE)
# - Fix PPO with LSTMs
# - PPO KL Penalty
# - DropoutPolicy
# - ACKTR

ALGO=ppo
ALGO=reinforce
N_STEPS=10000000
TEST_N_STEPS=100
NUM_WORKERS=3
DROPOUT=0.0

ENV=CartPole-v0
ENV=InvertedPendulum-v1
#ENV=InvertedDoublePendulum-v1
ENV=Ant-v1
#ENV=InvertedPendulumBulletEnv-v0
MODEL=fc
#MODEL=lstm
#MODEL=baseline

ifeq ($(ALGO),ppo)
    FREQ=2048
else
    FREQ=000
endif

ifeq ($(ENV),CartPole-v0)
    ifeq ($(MODEL),fc)
	LAYER_SIZE=128
	OPT=Adam
	LR=0.01
    endif
    ifeq ($(MODEL),lstm)
	LAYER_SIZE=32
	LR=0.0073
	OPT=SGD
    endif
endif

ifeq ($(ENV),InvertedPendulum-v1)
    ifeq ($(MODEL),fc)
	LAYER_SIZE=128
	LR=0.01
	OPT=Adam
    endif
    ifeq ($(MODEL),lstm)
	LAYER_SIZE=16
	LR=0.003
	OPT=SGD
    endif
endif

ifeq ($(ENV),InvertedDoublePendulum-v1)
    ifeq ($(MODEL),fc)
	LAYER_SIZE=64
	LR=0.01
	OPT=Adam
    endif
    ifeq ($(MODEL),lstm)
	LAYER_SIZE=16
	LR=0.003
	OPT=SGD
    endif
endif


ifeq ($(ENV),InvertedPendulumBulletEnv-v0)
    ifeq ($(MODEL),fc)
	LAYER_SIZE=128
	LR=0.01
	OPT=Adam
    endif
    ifeq ($(MODEL),lstm)
	LAYER_SIZE=16
	LR=0.001
	OPT=SGD
    endif
endif

ifeq ($(ENV),Ant-v1)
    ifeq ($(MODEL),fc)
	LAYER_SIZE=64
	LR=0.001
	OPT=Adam
    endif
    ifeq ($(MODEL),lstm)
	LAYER_SIZE=32
	LR=0.0033
	OPT=SGD
    endif
endif

ifeq ($(MODEL),baseline)
    LR=3e-4
    LR=3e-3
    OPT=Adam
    LAYER_SIZE=32
endif


.PHONY: all dev 

all: baseline

async:
	python async_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --model $(MODEL) --update_frequency $(FREQ) --max_path_length 50000

sync:
	python sync_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --model $(MODEL) --update_frequency $(FREQ) --max_path_length 50000

dev:
	python benchmark.py --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --model $(MODEL) --dropout $(DROPOUT) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --update_frequency $(FREQ) --max_path_length 100 --record True

baseline:
	python benchmark.py --algo ppo --env Reacher-v1 --n_steps $(N_STEPS) --model baseline --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr 3e-4 --layer_size 32 --update_frequency 2048 --max_path_length 50000


bench:
	python benchmark.py --algo $(ALGO) --env Ant-v1 --n_steps 250000 --model fc --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr 7e-4 --layer_size 64 --update_frequency 000 --max_path_length 50000
	python benchmark.py --algo $(ALGO) --env Ant-v1 --n_steps 250000 --model lstm --dropout $(DROPOUT) --n_test_iter 100 --opt SGD --lr 0.00073 --layer_size 32 --update_frequency 000 --max_path_length 50000
	python benchmark.py --algo $(ALGO) --env Ant-v1 --n_steps 250000 --model fc --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr 7e-4 --layer_size 128 --update_frequency 000 --max_path_length 50000
