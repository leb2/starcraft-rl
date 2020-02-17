#!/usr/bin/env bash

python -m run_agent \
--map TacticalRLTraining \
--save_name tactical3-nolstm \
--load_model \
--norender \
--step_mul 32 \
--parallel 8 \
--save_replay_episodes 0 \
--gamma 0.94 \
--td_lambda 0.94 \
--learning_rate 0.0001


