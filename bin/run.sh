#!/usr/bin/env bash

python -m run_agent2 \
--map TacticalRLTraining \
--save_name tactical11 \
--noload_model \
--norender \
--step_mul 24 \
--parallel 4 \
--save_replay_episodes 0 \
--gamma 0.94 \
--td_lambda 0.94 \
--learning_rate 0.0001

