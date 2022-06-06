#!/bin/bash
# Please call this from project root!

# clean up previous log file
rm -rf logs
mkdir -p logs

echo "running experiments..."

subset=(162 161 54 53 140 32 141 33 139 138 137 31 30 29 28 144 143 135 133 132)
(for i in "${subset[@]}"
do
	python -u ./predictive_exp.py -m regression randomforest lgbm -d $i -n 4 -w 1 -y 1 2>&1 | tee -a ./logs/log_job0.txt
done) &
(for i in "${subset[@]}"
do
	python -u ./predictive_exp.py -m nbeats nhits tcn tft -d $i -g true 2>&1 | tee -a ./logs/log_job1.txt
done) &
(for i in "${subset[@]}"
do
	python -u ./predictive_exp.py -m transformer rnn lstm gru -d $i -g true 2>&1 | tee -a ./logs/log_job2.txt
done)



echo "All Done!"


