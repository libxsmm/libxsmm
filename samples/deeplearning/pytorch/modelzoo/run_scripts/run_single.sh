#!/bin/bash

function print_vars {
  for VAR in ${!KMP_*} ${!OMP_*} LD_PRELOAD ${!DLRM_*} ${!PYTORCH_*} ${!PCL_*} ${!LIBXSMM_*} ${!EMULATE_*} VIRTUAL_ENV ${!ARGS_*} $@ ; do
    if ! test -z ${!VAR} ; then
       echo "Using $VAR=${!VAR}"
    fi
  done
}

while (( "$#" )); do
  case "$1" in
    -n|-np)
      ARGS_NTASKS=$2
      shift 2
      ;;
    -ppn)
      ARGS_PPN=$2
      shift 2
      ;;
    --) # end argument parsing
      shift
      break
      ;;
    -*|--*=) # unsupported flags
      echo "Error: Unsupported flag $1" >&2
      exit 1
      ;;
    *) # preserve positional arguments
      break
      ;;
  esac
done

if ! test -z $SLURM_JOB_ID ; then
  PREFIX="srun -n 1 -N 1 "
else
  PREFIX=
fi

echo "Running $NP tasks on $NNODES nodes"

NUM_THREADS=`$PREFIX lscpu | grep "Core(s) per socket" | awk '{print $NF}'`
NUM_SOCKETS=`$PREFIX lscpu | grep "Socket(s):" | awk '{print $NF}'`
NUM_NUMA_NODES=`$PREFIX lscpu | grep "NUMA node(s):" | awk '{print $NF}'`
THREADS_PER_CORE=`$PREFIX lscpu | grep "Thread(s) per core:" | awk '{print $NF}'`
PHYSCPU=0-$(( NUM_THREADS - 1 ))

export OMP_NUM_THREADS=$(( NUM_THREADS ))
export KMP_AFFINITY=compact,1,granularity=fine,verbose
export KMP_BLOCKTIME=1
export LD_PRELOAD=${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so

which python icc gcc  2> /dev/null

echo "#### INITIAL ENV ####"
print_vars
echo "#### INITIAL ENV ####"

echo "PyTorch version: `python -c "import torch; print(torch.__version__)" 2> /dev/null`"

if ! test -z $SLURM_JOB_ID ; then
srun hostname | sort -u
else
hostname
fi

CMD=$1
shift
ARGS="$@"

EXE_ARGS="$PREFIX numactl -m 0 -C ${PHYSCPU} "

#echo "Running mpiexec.hydra ${MPIEXE_ARGS} $CMD $@"
eval set -- "${EXE_ARGS} $CMD $ARGS"
echo "Running $@"
echo "Start Time:  `date`"
$@
echo "End Time:    `date`"

