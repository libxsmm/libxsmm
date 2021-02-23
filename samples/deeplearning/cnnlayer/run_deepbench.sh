#!/usr/bin/env bash
set -eo pipefail

UNAME=$(command -v uname)
SORT=$(command -v sort)
GREP=$(command -v grep)
CUT=$(command -v cut)
WC=$(command -v wc)
TR=$(command -v tr)

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1000; fi
else # check
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 6 ]
then
  echo "Usage: $(basename $0) iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values: 1000 1 f32 A L 1"
  ITERS=${CHECK_DNN_ITERS}
  NUMA=-1
  BIN=f32
  TYPE="A"
  FORMAT="L"
  PAD=1
else
  ITERS=$1
  NUMA=$2
  BIN=$3
  TYPE=$4
  FORMAT=$5
  PAD=$6
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

if [ "" = "${LIBXSMM_TARGET_HIDDEN}" ] || [ "0" = "${LIBXSMM_TARGET_HIDDEN}" ]; then
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} NUMACTL=\"${NUMACTL}\""
  echo
fi

# ./layer_example_${BIN} iters inpWidth inpHeight nImg nIfm nOfm kw kh padw padh stride type
#
if [ "${BIN}" != "f32" ]; then
  true
else
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  700 161  4     1   32  20 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
fi
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  700 161  8     1   32  20 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  700 161 16     1   32  20 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  700 161 32     1   32  20 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  341  79  4    32   32  10 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  341  79  8    32   32  10 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  341  79 16    32   32  10 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  341  79 32    32   32  10 5 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  480  48 16     1   16   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  240  24 16    16   32   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  120  12 16    32   64   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   60   6 16    64  128   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  108 108  8     3   64   3 3 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   54  54  8    64   64   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   27  27  8   128  128   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8   128  256   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7  8   256  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  224 224  8     3   64   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  112 112  8    64  128   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56  8   128  256   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28  8   256  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8   512  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7  8   512  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  224 224 16     3   64   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  112 112 16    64  128   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56 16   128  256   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   256  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   512  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16   512  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  224 224 16     3   64   7 7 3 3 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   192   32   5 5 2 2 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   192   64   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   512   48   5 5 2 2 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   512  192   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16   832  256   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16   832  128   5 5 2 2 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56  8    64   64   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56  8    64  256   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28  8   128  128   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28  8   128  512   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8   256  256   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8   256  256   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8   256 1024   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7  8   512  512   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7  8  2048  512   1 1 3 3 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56 16    64   64   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56 16    64  256   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   128  128   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   128  512   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   256  256   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   256  256   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   256 1024   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16   512  512   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16  2048  512   1 1 3 3 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  700 161 16     1   64   5 5 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  350  80 16    64   64   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  350  80 16    64  128   5 5 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  175  40 16   128  128   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  175  40 16   128  256   5 5 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   84  20 16   256  256   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   84  20 16   256  512   5 5 1 1 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   42  10 16   512  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  112 112  8    64   64   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56  8    64  256   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56  8   256   64   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56  8   256  128   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28  8   128  512   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28  8   512  128   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28  8   512  256   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8   256 1024   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28  8   512 1024   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8  1024  256   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8   256 1024   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8  1024  512   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7  8   512  512   3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7  8   512 2048   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14  8  1024 2048   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7  8  2048  512   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}  112 112 16    64   64   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56 16    64  256   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56 16   256   64   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   56  56 16   256  128   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   128  512   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   512  128   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   512  256   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   256 1024   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   28  28 16   512 1024   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16  1024  256   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16   256 1024   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16  1024  512   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16   512 512    3 3 1 1 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16   512 2048   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}   14  14 16  1024 2048   1 1 0 0 2 ${TYPE} ${FORMAT} ${PAD}
${NUMACTL}  ./layer_example_${BIN}  ${ITERS}    7   7 16  2048  512   1 1 0 0 1 ${TYPE} ${FORMAT} ${PAD}

