#!/bin/bash
#$ -l h_rt=8:00:00  #time needed
#$ -pe smp 6 #number of cores
#$ -l rmem=2G #number of memory
#$ -o master_tuning_all.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M mkrivova1@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

cd /data/acq18mk/
module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
source activate myspark

spark-submit --driver-memory 32G /home/acq18mk/master/short_notebooks/6_parameter_tuning_all.py
