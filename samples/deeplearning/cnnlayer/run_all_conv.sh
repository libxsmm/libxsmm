#!/usr/bin/env bash
set -eo pipefail

UNAME=$(command -v uname)
SORT=$(command -v sort)
GREP=$(command -v grep)
CUT=$(command -v cut)
WC=$(command -v wc)
TR=$(command -v tr)

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_MB}" ]; then CHECK_DNN_MB=64; fi
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 7 ]
then
  echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 64 1000 1 f32 A L 1"
  MB=${CHECK_DNN_MB}
  ITERS=${CHECK_DNN_ITERS}
  NUMA=-1
  BIN=f32
  TYPE="A"
  FORMAT="L"
  PAD=1
else
  MB=$1
  ITERS=$2
  NUMA=$3
  BIN=$4
  TYPE=$5
  FORMAT=$6
  PAD=$7
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

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh padw padh stride type
#
# ResNet50
${NUMACTL} ./layer_example_${BIN} ${ITERS} 224 224 ${MB} 3 64 7 7 3 3 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 64 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 64 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 64 64 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 256 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 256 512 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 256 128 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 128 128 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 128 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 512 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 512 1024 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 512 256 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 256 256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 256 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 1024 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 1024 2048 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 1024 512 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 512 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 512 2048 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 2048 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
# fastrcnn
${NUMACTL} ./layer_example_${BIN} ${ITERS} 606 756 ${MB} 3 64 7 7 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 150 188 ${MB} 64 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 150 188 ${MB} 64 64 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 150 188 ${MB} 64 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 150 188 ${MB} 256 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 152 190 ${MB} 64 64 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 75 94 ${MB} 64 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 75 94 ${MB} 256 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 75 94 ${MB} 256 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 75 94 ${MB} 128 128 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 75 94 ${MB} 128 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 75 94 ${MB} 512 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 77 96 ${MB} 128 128 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 128 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 512 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 256 256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 256 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 512 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 1024 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 1024 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 512 48 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 47 ${MB} 512 24 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 1024 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 512 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 512 2048 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 1024 2048 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 2048 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
#maskrcnn
${NUMACTL} ./layer_example_${BIN} ${ITERS} 1030 1030 ${MB} 3 64 7 7 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 64 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 64 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 64 64 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 256 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 256 128 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 256 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 256 512 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 128 128 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 128 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 512 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 512 256 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 256 256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 256 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 512 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 512 1024 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 1024 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 1024 2048 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 1024 512 1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 512 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 512 2048 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 2048 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 2048 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 256 256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 256 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 512 6 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 32 32 ${MB} 512 12 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 16 16 ${MB} 256 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 16 16 ${MB} 512 6 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 16 16 ${MB} 512 12 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 256 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 512 6 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 64 64 ${MB} 512 12 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 256 256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 256 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 512 6 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 128 128 ${MB} 512 12 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 256 256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 256 512 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 512 6 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 256 256 ${MB} 512 12 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 256 1024 7 7 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 1 1 ${MB} 1024 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 256 256 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 256 256 2 2 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 256 81 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
#xception
${NUMACTL} ./layer_example_${BIN} ${ITERS} 299 299 ${MB} 3 32 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 149 149 ${MB} 32 64 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 147 147 ${MB} 64 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 147 147 ${MB} 128 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 74 74 ${MB} 128 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 74 74 ${MB} 256 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 37 37 ${MB} 256 728 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 37 37 ${MB} 728 728 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 19 19 ${MB} 728 728 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 19 19 ${MB} 728 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 10 10 ${MB} 1024 1536 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 10 10 ${MB} 1536 2048 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
#yolov2
${NUMACTL} ./layer_example_${BIN} ${ITERS} 610 610 ${MB} 3 32 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 306 306 ${MB} 32 64 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 154 154 ${MB} 64 128 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 152 152 ${MB} 128 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 78 78 ${MB} 128 256 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 76 76 ${MB} 256 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 40 40 ${MB} 256 512 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 38 ${MB} 512 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 21 21 ${MB} 512 1024 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 19 19 ${MB} 1024 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 21 21 ${MB} 1024 1024 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 38 38 ${MB} 512 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 21 21 ${MB} 1280 1024 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 19 19 ${MB} 1024 425 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
#mobilenet
${NUMACTL} ./layer_example_${BIN} ${ITERS} 224 224 ${MB} 3 32 3 3 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 112 112 ${MB} 32 64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 64 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 56 56 ${MB} 128 128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 128 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 28 28 ${MB} 256 256 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 256 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 14 14 ${MB} 512 512 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 512 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 7 7 ${MB} 1024 1024 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 1 1 ${MB} 1024 5 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
#AlexNet
${NUMACTL} ./layer_example_${BIN} ${ITERS} 227 227  ${MB}    3   64 11 11 0 0 4 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  27  27  ${MB}   64  192  5  5 2 2 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  13  13  ${MB}  192  384  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  13  13  ${MB}  384  256  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  13  13  ${MB}  256  256  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
#Overfeat
${NUMACTL} ./layer_example_${BIN} ${ITERS} 231 231  ${MB}    3   96 11 11 0 0 4 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}   96  256  5  5 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  12  12  ${MB}  256  512  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  12  12  ${MB}  512 1024  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  12  12  ${MB} 1024 1024  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
#Googlenetv1
${NUMACTL} ./layer_example_${BIN} ${ITERS}   224  224  ${MB}     3    64  7  7  3  3  2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    56   56  ${MB}    64   192  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    96  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    96   128  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    16  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    16    32  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   192    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   128   192  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}    32    96  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    28   28  ${MB}   256    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480   192  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    96  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    96   208  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    16  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    16    48  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   480    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   160  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   112  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   112   224  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    64  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   128   256  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512   144  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   144   288  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   512    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32    64  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   256  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   160  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   160   320  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}    32   128  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}    14   14  ${MB}   528   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   256  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   160  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   160   320  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    32  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    32   128  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   128  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   384  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832   192  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   192   384  3  3  1  1  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}   832    48  1  1  0  0  1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}     7    7  ${MB}    48   128  5  5  2  2  1 ${TYPE} ${FORMAT} ${PAD}
#Googlenetv3
${NUMACTL} ./layer_example_${BIN} ${ITERS}  299 299 ${MB}     3   32 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  149 149 ${MB}    32   32 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  147 147 ${MB}    32   64 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  73  73  ${MB}    64   80 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  73  73  ${MB}    80  192 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  71  71  ${MB}    80  192 3 3 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   192   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}    64   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}    96   96 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   192   48 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}    48   64 5 5 2 2 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   192   32 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   256   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   256   48 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   288   64 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   288   48 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}    96   96 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  35  35  ${MB}   288  384 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   768  128 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   128  128 1 7 0 3 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   128  128 7 1 3 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   128  192 7 1 3 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   128  192 1 7 0 3 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   768  192 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   768  160 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   160  160 1 7 0 3 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   160  160 7 1 3 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   160  192 7 1 3 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   160  192 1 7 0 3 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   192  192 1 7 0 3 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   192  192 7 1 3 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   192  192 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  17  17  ${MB}   192  320 3 3 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  1280  320 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  1280  192 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  1280  448 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}   448  384 3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}   384  384 1 3 0 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}   384  384 3 1 1 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  1280  384 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  2048  320 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  2048  192 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  2048  448 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}  2048  384 1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  8   8   ${MB}   384  384 1 3 0 1 1 ${TYPE} ${FORMAT} ${PAD}
#vgga
${NUMACTL} ./layer_example_${BIN} ${ITERS} 224 224  ${MB}    3   64  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS} 112 112  ${MB}   64  128  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  128  256  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  56  56  ${MB}  256  256  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  256  512  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  28  28  ${MB}  512  512  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL} ./layer_example_${BIN} ${ITERS}  14  14  ${MB}  512  512  3  3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
