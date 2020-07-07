#!/bin/bash
#$ -P rse-com6012
#$ -q rse-com6012.q
#$ -l h_rt=2:00:00  #time needed
#$ -pe smp 6 #number of cores
#$ -l rmem=6G #number of memory
#$ -o master_tuning.output #This is where your output and errors are logged.
#$ -j y # normal and error outputs into a single file (the file above)
#$ -M mkrivova1@sheffield.ac.uk #Notify you by email, remove this line if you don't like
#$ -m ea #Email you when it finished or aborted
#$ -cwd # Run job from current directory

cd /data/acq18mk/
module load apps/java/jdk1.8.0_102/binary
module load apps/python/conda
source activate myspark

spark-submit --driver-memory 64G /home/acq18mk/master/short_notebooks/5_3_parameter_tuning.py