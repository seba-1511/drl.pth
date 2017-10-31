
# TODO:
# - Cleanup PPO
# - Train A2C on Pong
# - Add KL Penalty (PPO)
# - Learned and not learned StateNormalizer, as part of policy
# - Include Adaptive scaling of KL penaly (PPO)
# - Fix PPO with LSTMs
# - PPO KL Penalty
# - DropoutPolicy
# - ACKTR

ALGO=ppo
#ALGO=reinforce
N_STEPS=10000
TEST_N_STEPS=100
NUM_WORKERS=4
DROPOUT=0.0

ENV=CartPole-v0
ENV=InvertedPendulum-v1
#ENV=InvertedDoublePendulum-v1
ENV=Ant-v1
ENV=DiscreteOrientation-v0
#ENV=AntBulletEnv-v0
#ENV=InvertedPendulumBulletEnv-v0
MODEL=fc
#MODEL=lstm
MODEL=baseline

ifeq ($(ALGO),ppo)
    FREQ=2048
else
    FREQ=005
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
	LAYER_SIZE=64
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


ifeq ($(ENV),AntBulletEnv-v0)
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
    LR=3e-3
    OPT=Adam
    LAYER_SIZE=32
endif

#ENV=Reacher-v1

.PHONY: all dev 

all: baseline

async:
	python examples/async_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --model $(MODEL) --update_frequency $(FREQ) --max_path_length 5000

sync:
	python examples/sync_bench.py --n_proc $(NUM_WORKERS) --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --model $(MODEL) --update_frequency $(FREQ) --max_path_length 5000

dev:
	python examples/benchmark.py --algo $(ALGO) --env $(ENV) --n_steps $(N_STEPS) --model $(MODEL) --dropout $(DROPOUT) --n_test_iter 100 --opt $(OPT) --lr $(LR) --layer_size $(LAYER_SIZE) --update_frequency $(FREQ) --max_path_length 100 --record True

baseline:
	python examples/benchmark.py --algo ppo --env Reacher-v1 --n_steps $(N_STEPS) --model baseline --dropout $(DROPOUT) --n_test_iter 100 --opt Adam --lr 3e-3 --layer_size 32 --update_frequency 2048 --max_path_length 5000

hl:
	python examples/high_level.py --n_steps 1000000 --max_path_length 5000 --update_frequency 1000 --print_interval 1000 --record True
