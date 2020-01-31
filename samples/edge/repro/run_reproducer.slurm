#!/usr/bin/env bash
#----------------------------------------------------
# Sample SLURM job script
#   for TACC Stampede2 KNL nodes
#
#   *** Serial Job on Normal Queue ***
#
# Last revised: 27 Jun 2017
#
# Notes:
#
#   -- Copy/edit this script as desired.  Launch by executing
#      "sbatch knl.serial.slurm" on a Stampede2 login node.
#
#   -- Serial codes run on a single node (upper case N = 1).
#        A serial code ignores the value of lower case n,
#        but slurm needs a plausible value to schedule the job.
#
#   -- For a good way to run multiple serial executables at the
#        same time, execute "module load launcher" followed
#        by "module help launcher".

#----------------------------------------------------

#SBATCH -J reproducer      # Job name
#SBATCH -o output/reproducer.o%j  # Name of stdout output file
#SBATCH -e output/reproducer.e%j  # Name of stderr error file
#SBATCH -p flat-quadrant      # Queue (partition) name
#SBATCH -N 1               # Total # of nodes (must be 1 for serial)
#SBATCH -n 1               # Total # of mpi tasks (should be 1 for serial)
#SBATCH --cpus-per-task=272
#SBATCH -t 01:30:00        # Run time (hh:mm:ss)
#SBATCH -A TG-MCA03S012    # Allocation name (req'd if you have more than 1)
#SBATCH --exclusive        # X mode

# Other commands must follow all #SBATCH directives...
####XSBATCH -p flat-quadrant   # Queue (partition) name

#module list
cd $(pwd)
#date

# Launch serial code...

date

export MATS_DIR=./mats_padded
export NELMTS=82500

export OMP_NUM_THREADS=66
export KMP_AFFINITY=verbose,granularity=fine,compact,1,2
#export KMP_AFFINITY=proclist=[`seq -s',' 4 5`],explicit
export KMP_HW_SUBSET=1T

export RATIO=150
export OSTEPS=(75 30 10 5 2 1)
for prc_cfr in 32_16 64_8
do
  for odr in `seq 2 7`
  do
    export NSTEPS=$((${OSTEPS[$((odr-2))]}*RATIO))
    numactl -m 1 ./bin/local_${odr}_${prc_cfr} ${NSTEPS} ${NELMTS} > output/local_${odr}_${prc_cfr}.log
    #./bin/local_${odr}_${prc_cfr} ${NSTEPS} ${NELMTS} > output/local_${odr}_${prc_cfr}.log
  done
done

export RATIO=100
export OSTEPS=(75 30 14 8 4 2)
for prc_cfr in 32_16 64_8
do
  for odr in `seq 2 7`
  do
    export NSTEPS=$((${OSTEPS[$((odr-2))]}*RATIO))
    numactl -m 1 ./bin/neigh_${odr}_${prc_cfr} ${NSTEPS} ${NELMTS} > output/neigh_${odr}_${prc_cfr}.log
    #./bin/neigh_${odr}_${prc_cfr} ${NSTEPS} ${NELMTS} > output/neigh_${odr}_${prc_cfr}.log
  done
done

export RATIO=50
export OSTEPS=(75 30 10 8 4 2)
for prc_cfr in 32_16 64_8
do
  for odr in `seq 2 7`
  do
    export NSTEPS=$((${OSTEPS[$((odr-2))]}*RATIO))
    numactl -m 1 ./bin/combine_${odr}_${prc_cfr} ${NSTEPS} ${NELMTS} > output/combine_${odr}_${prc_cfr}.log
  done
done


date

# ---------------------------------------------------
