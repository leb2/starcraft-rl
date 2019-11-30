#!/usr/bin/env bash

python -m run_agent2 \
--map DefeatRoaches \
--save_name roaches-with-stalkers2 \
--load_model \
--norender \
--step_mul 8 \
--parallel 10 \
--save_replay_episodes 0 \
--gamma 0.95 \
--td_lambda 0.95 \
--learning_rate 0.0001

