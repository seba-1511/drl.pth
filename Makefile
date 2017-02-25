
.PHONY: all dev

all:
	python benchmark.py --algo reinforce --env InvertedPendulum-v1 --n_iter 400 --n_test_iter 10 --timesteps_per_batch 5000 --opt Adam --lr 0.001
