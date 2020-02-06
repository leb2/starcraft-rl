#!/usr/bin/env bash

python -m run_agent2 \
--map BuildMarines \
--save_name buildmarines1 \
--load_model \
--norender \
--step_mul 64 \
--parallel 8 \
--save_replay_episodes 0 \
--gamma 0.92 \
--td_lambda 0.92 \
--learning_rate 0.0001

