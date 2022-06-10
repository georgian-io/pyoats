#!/bin/bash
# Please call this from project root!

# clean up previous log file
rm -rf logs
mkdir -p logs

echo "running experiments..."

subset=(162 161 54 53)
#(for i in "${subset[@]}"
#do
#	python -u ./predictive_exp.py -m regression randomforest lgbm -d $i -n 4 2>&1 | tee -a ./logs/log_job0.txt
#done) &
(for i in "${subset[@]}"
do
	python -u ./predictive_exp.py -m nhits tcn -d $i -g true -y 10 -w 10 2>&1 | tee -a ./logs/log_job1.txt
done) 


echo "All Done!"


