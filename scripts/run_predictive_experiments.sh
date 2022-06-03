#!/bin/sh
# Please call this from project root!

# clean up previous log file
rm -rf logs
mkdir -p logs

echo "running experiments..."

# Regression + RandomForest + LGBM -> 5 cpus
# DL Model run at the same time in pairs

python -u ./predictive_exp.py -m regression randomforest lgbm -n 4 2>&1 | tee -a ./logs/log_job0.txt &
python -u ./predictive_exp.py -m nbeats nhits -g true 2>&1 | tee -a ./logs/log_job1.txt &
python -u ./predictive_exp.py -m tcn tft -g true 2>&1 | tee -a ./logs/log_job2.txt &
python -u ./predictive_exp.py -m transformer rnn -g true 2>&1 | tee -a ./logs/log_job3.txt &
python -u ./predictive_exp.py -m lstm gru -g true 2>&1 | tee -a ./logs/log_job4.txt

echo "All Done!"


