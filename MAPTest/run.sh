#PBS -l walltime=04:00:00
#PBS -m abe
#PBS -M s103407@student.dtu.dk

module load python

source ./.env/bin/activate

cd $PBS_O_WORKDIR

python map.py