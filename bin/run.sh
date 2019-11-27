#!/usr/bin/env bash

python -m run_agent2 \
--map MoveToBeacon \
--save_name beacon-orig3 \
--noload_model \
--norender \
--step_mul 8 \
--parallel 8 \
--save_replay_episodes 0 \
--gamma 0.95 \
--td_lambda 0.95 \
--learning_rate 0.0001