#!/usr/bin/env bash
set -eo pipefail

UNAME=$(command -v uname)
SORT=$(command -v sort)
GREP=$(command -v grep)
CUT=$(command -v cut)
WC=$(command -v wc)
TR=$(command -v tr)
NUMA=-1

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=100; fi
else # check
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 7 ]
then
  N=${CHECK_DNN_MB}
  ITERS=${CHECK_DNN_ITERS}
  NB=32
  CB=256
  KB=256
  NNB=16
  FRAC=0.9
else
  N=$1
  ITERS=$2
  NB=$3
  CB=$4
  KB=$5
  NNB=$6
  FRAC=$7
fi

if [ "${GREP}" ] && [ "${SORT}" ] && [ "${CUT}" ] && [ "${TR}" ] && [ "${WC}" ]; then
  if [ "$(command -v lscpu)" ]; then
    NS=$(lscpu | ${GREP} -m1 "Socket(s)" | ${TR} -d " " | ${CUT} -d: -f2)
    if [ "" = "${NS}" ]; then NS=1; fi
    NC=$((NS*$(lscpu | ${GREP} -m1 "Core(s) per socket" | ${TR} -d " " | ${CUT} -d: -f2)))
    NT=$((NC*$(lscpu | ${GREP} -m1 "Thread(s) per core" | ${TR} -d " " | ${CUT} -d: -f2)))
  elif [ -e /proc/cpuinfo ]; then
    NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
    if [ "" = "${NS}" ] || [ "" = "${NS}" ]; then NS=1; fi
    NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
    NT=$(${GREP} "core id" /proc/cpuinfo  | ${WC} -l | ${TR} -d " ")
  elif [ "Darwin" = "$(uname)" ]; then
    NS=$(sysctl hw.packages    | ${CUT} -d: -f2 | ${TR} -d " ")
    NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | ${TR} -d " ")
    NT=$(sysctl hw.logicalcpu  | ${CUT} -d: -f2 | ${TR} -d " ")
  fi
  if [ "${NC}" ] && [ "${NT}" ]; then
    HT=$((NT/NC))
  else
    NS=1 NC=1 NT=1 HT=1
  fi
  if [ "$(command -v numactl)" ]; then
    NN=$(numactl -H | ${GREP} "available:" | ${CUT} -d' ' -f2)
  else
    NN=${NS}
  fi
fi

CPUFLAGS=$(if [ "${GREP}" ] && [ "${CUT}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | ${CUT} -d: -f2- || true; fi)
if [ "${GREP}" ] && [ "$(echo "${CPUFLAGS}" | ${GREP} -o avx512er)" ]; then
  if [ "0" != "$((0>NUMA))" ] && [ "0" != "$((NS<NN))" ]; then
    NUMACTL="numactl --preferred=${NS} ${TOOL_COMMAND}"
  elif [ "0" != "$((0<=NUMA && NUMA<NN))" ]; then
    NUMACTL="numactl --preferred=${NUMA} ${TOOL_COMMAND}"
  elif [ "1" != "${NS}" ]; then
    #NUMACTL="numactl -i all ${TOOL_COMMAND}"
    NUMACTL="${TOOL_COMMAND}"
  fi
else
  NUMACTL="${TOOL_COMMAND}"
fi

if [ "" = "${OMP_NUM_THREADS}" ] || [ "0" = "${OMP_NUM_THREADS}" ]; then
  if [ "" = "${KMP_AFFINITY}" ]; then
    export KMP_AFFINITY=compact,granularity=fine KMP_HW_SUBSET=1T
  fi
  export OMP_NUM_THREADS=$((NC))
fi

if [ "" = "${MB}" ] || [ "0" = "${MB}" ]; then
  MB=${OMP_NUM_THREADS}
fi

if [ "" = "${LIBXSMM_TARGET_HIDDEN}" ] || [ "0" = "${LIBXSMM_TARGET_HIDDEN}" ]; then
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} NUMACTL=\"${NUMACTL}\""
  echo
fi

# ./parallel_sparse_weight_B_conv nImg inpHeight inpWidth nIfm nOfm kh kw padh padw strideh stridew NB CB KB NNB sprase-fract Iters
#
#${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  224 224    3   64 7 7 3 3 2 2 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  56  56    64  256 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  56  56    64   64 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  56  56    64   64 3 3 1 1 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  56  56   256   64 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  56  56   256  512 1 1 0 0 2 2 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  56  56   256  128 1 1 0 0 2 2 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  28  28   128  128 3 3 1 1 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  28  28   128  512 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  28  28   512  128 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  28  28   512 1024 1 1 0 0 2 2 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  28  28   512  256 1 1 0 0 2 2 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  14  14   256  256 3 3 1 1 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  14  14   256 1024 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  14  14  1024  256 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  14  14  1024 2048 1 1 0 0 2 2 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}  14  14  1024  512 1 1 0 0 2 2 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}   7   7   512  512 3 3 1 1 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}   7   7   512 2048 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}
${NUMACTL} ./parallel_sparse_weight_B_conv ${N}   7   7  2048  512 1 1 0 0 1 1 ${NB} ${CB} ${KB} ${NNB} ${FRAC} ${ITERS}

