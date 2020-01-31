#!/usr/bin/env sh

# launch.sh [ARCH] [NUM_NODES] [TOPOLOGY] [MLSL_VER]
# for example: launch.sh knl 2 googlenet ep

#trap "set +x; sleep 1; set -x" DEBUG

CUR_DIR=$(dirname `which $0`)

if [ -z $1 ] || [ -z $2 ] || [ -z $3 ]; then
    echo "use: launch.sh [ARCH] [NUM_PROCS] [TOPOLOGY] [MLSL_VER](optional)"
    exit 1
fi

arch_=$1
numprocs=$2
topo=$3
mlslver=ep #${4:-"ep"}
TRAIN_VAL_PROTOTXT=`readlink -f $4`
SOLVER_PROTOTXT=`readlink -f $5`
checkpoint=$6

export EPLIB_SHM_SIZE_GB=10
export MLSL_HEAP_SIZE_GB=10

mcdram=1
export CLUSTER=endv
echo "Running $topo topology with mlsl $mlslver on $arch_ in $numprocs processes on $CLUSTER cluster"

#source ${CUR_DIR}/setup_env.sh

#${CUR_DIR}/split-train-solver $arch_ $numprocs $topo

if [ ! -d "${CUR_DIR}/$arch_" ]; then
    mkdir ${CUR_DIR}/$arch_
fi

# Create a new directory for each run and copy the required input files
work_dir="${CUR_DIR}/${arch_}/${arch_}_${numprocs}_${topo}"
if [ $checkpoint == 0 ]; then
  rm -rf $work_dir
  mkdir -p $work_dir
fi
cd $work_dir
if [ $checkpoint == 0 ]; then
  if [ ! -d weights ]; then
    mkdir -p weights
  fi
  if [ ! -d weights30 ]; then
    mkdir -p weights30
  fi
  if [ ! -d weights60 ]; then
    mkdir -p weights60
  fi
  if [ ! -d weights80 ]; then
    mkdir -p weights80
  fi
fi
export WORK_DIR=$work_dir

# Store all node names in an array.
# Later we go thru this array # Note: PBS_NODEFILE is set by lsf, based on the parameters we pass
# in bsub. For instance, if we request 2 nodes with bsub, PBS_NODEFILE contains two host names

if [ "$CLUSTER" == "endv" ]; then
    cat $PBS_NODEFILE | uniq|sort > ${CUR_DIR}/hostfile
elif [ "$CLUSTER" == "pcl" ]; then
    scontrol show hostnames > ${CUR_DIR}/hostfile
fi

if [ ! -f "${CUR_DIR}/hostfile" ]; then
    echo "Create hostfile at first"
    exit 1
fi

# Names to configfile, binary (executable) files #
cfile=${WORK_DIR}/nodeconfig.txt
GXM_PATH=${CUR_DIR}
xeonbin="${GXM_PATH}/build/bin/gxm train"
cpuhostfile=${CUR_DIR}/hostfile
nodenames=( `cat ${cpuhostfile}` )

# EPLIB configuration

if [ ${arch_} == skx ]; then
    numservers=2
    listep=6,34
elif [ ${arch_} == clx ]; then
    numservers=2
    listep=6,34
elif [ ${arch_} == clxap ]; then
    numservers=4
    listep=6,30,54,78
elif [ ${arch_} == knl ]; then
    numservers=2
    listep=6,7,8,9,10,11,12,13
elif [ ${arch_} == knm ]; then
    numservers=2
    listep=6,7,8,9,10,11,12,13
fi

threadspercore=1
ppncpu=1

maxcores=`cpuinfo | grep "Processors(CPUs)" | awk '{print $3}'`
maxcores=`cpuinfo | grep "Cores             :" | awk '{print $3}'`
load_bal_threads=0
numthreads=$(((maxcores-numservers-load_bal_threads)*threadspercore))
#numthreads=32

# MLSL configuration
export MLSL_LOG_LEVEL=1
export MLSL_NUM_SERVERS=${numservers}
export MLSL_SERVER_AFFINITY="${listep}"

# PSM2 configuration
export PSM2_MQ_RNDV_HFI_WINDOW=2097152 # to workaround PSM2 bug in IFS 10.2 and 10.3
export PSM2_IDENTIFY=1 # for debug
export HFI_NO_CPUAFFINITY=1

# IMPI configuration
export I_MPI_FABRICS=tmi
export I_MPI_TMI_PROVIDER=psm2
export I_MPI_FALLBACK=0
export I_MPI_DYNAMIC_CONNECTION=0
export I_MPI_SCALABLE_OPTIMIZATION=0
export I_MPI_PIN_MODE=lib
export I_MPI_PIN_DOMAIN=node
export I_MPI_DEBUG=6

#export MKL_CBWR=AUTO

# Produce the configuration file for mpiexec. Each line of the config file contains a # host, enviornment, binary name.
rm -f $cfile
node_id=0
numnodes=( `cat ${cpuhostfile} | grep -v ^$ | wc -l` )
max_ppn=$((numprocs/numnodes))
numthreads_per_proc=$((numthreads/max_ppn))
#MPIEXECARGS=" -np ${numnodes} -ppn $max_ppn -genv MLSL_NUM_SERVERS ${numservers} -genv MLSL_SERVER_AFFINITY \"${listep}\" -genv OMP_NUM_THREADS ${numthreads_per_proc} -genv KMP_AFFINITY \"$affinitystr\" "

# OMP configuration
if [ "$threadspercore" == "1" ]; then
  if [ "$numservers" == "0" ]; then
    affinitystr="proclist=[0-$((maxcores-1))],granularity=thread,explicit"
  elif [ "$numservers" == "2" ]; then
    affinitystr="proclist=[0-5,7-33,35-55],granularity=thread,explicit"
    #affinitystr="proclist=[0-5,7-16,28-33,35-44],granularity=thread,explicit"
  elif [ "$numservers" == "1" ]; then
    affinitystr="proclist=[0-5,7-27],granularity=thread,explicit"
  elif [ "$numservers" == "4" ]; then
    affinitystr="proclist=[0-5,7-29,31-53,55-77,79-95],granularity=thread,explicit"
  fi
else
    affinitystr="proclist=[0-5,$((5+numservers+1))-$((maxcores-1)),$((maxcores))-$((maxcores+5)),$((maxcores+5+numservers+1))-$((2*maxcores-1))],granularity=thread,explicit"
fi
export KMP_AFFINITY=$affinitystr

echo THREAD SETTINGS: Affinity $affinitystr Threads $numthreads Placement $KMP_PLACE_THREADS

MPIEXECARGS=" -np ${numnodes} -ppn $max_ppn -genv OMP_NUM_THREADS ${numthreads_per_proc} -genv KMP_AFFINITY \"$affinitystr\" "
mkdir -p ${WORK_DIR}/${arch_}_${numprocs}_${topo}
if [ ${arch_} == skx ]; then
    export NUMACTLCMD=
elif [ ${arch_} == skxbf16 ]; then
    export NUMACTLCMD=
elif [ ${arch_} == clx ]; then
    export NUMACTLCMD=
elif [ ${arch_} == clxap ]; then
    export NUMACTLCMD=
elif [ ${arch_} == knl ]; then
    export NUMACTLCMD="numactl --preferred=$mcdram"
elif [ ${arch_} == knm ]; then
    export NUMACTLCMD="numactl --preferred=$mcdram"
fi

cd $WORK_DIR

if [ "$mlslver" == "ep" ]; then
    for host in `cat $cpuhostfile`; do
        ssh -n $host "rm -rf /dev/shm/*shm*; killall -q mpiexec.hydra pcldnn_server; killall -q mpiexec.hydra ep_server; for j in \$(ipcs -a | awk '{print \$1}' | grep -v '\-' | grep -v 'key'); do ipcrm -M \${j} > /dev/null 2&>1; done" &
    done
    wait
fi

if [ "$mlslver" == "nompi" ]; then
    echo "nompi"
    $NUMACTLCMD $xeonbin ${TRAIN_VAL_PROTOTXT} ${SOLVER_PROTOTXT}
else
    echo "mpiexec"
    echo GLOG_minloglevel=0 mpiexec.hydra -l $MPIEXECARGS -hostfile $cpuhostfile $NUMACTLCMD $xeonbin ${TRAIN_VAL_PROTOTXT} ${SOLVER_PROTOTXT}
    GLOG_minloglevel=0 mpiexec.hydra -l $MPIEXECARGS -hostfile $cpuhostfile $NUMACTLCMD $xeonbin ${TRAIN_VAL_PROTOTXT} ${SOLVER_PROTOTXT}  2>&1 | tee -a outputCluster.txt
fi
