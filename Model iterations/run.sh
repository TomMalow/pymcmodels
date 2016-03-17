#PBS -l walltime=06:00:00
#PBS -m abe
#PBS -M s103407@student.dtu.dk

module load python

source ~/.env/bin/activate

cd $PBS_O_WORKDIR

python GibbsVSMH.py