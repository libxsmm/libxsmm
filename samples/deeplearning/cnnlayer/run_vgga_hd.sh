#!/usr/bin/env bash
set -eo pipefail

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=100; fi
else # check
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 7 ]; then
  if [ $# -ne 0 ]; then
    echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 128 1000 1 f32 A L 1"
  fi
  MB=1
  ITERS=${CHECK_DNN_ITERS}
  NUMA=-1
  TYPE=f32
  KIND=F
  FORMAT=L
  PADMODE=1
else
  MB=$1
  ITERS=$2
  NUMA=$3
  TYPE=$4
  KIND=$5
  FORMAT=$6
  PADMODE=$7
fi

UNAME=$(command -v uname)
PASTE=$(command -v paste)
GREP=$(command -v grep)
SORT=$(command -v sort)
DATE=$(command -v date)
CUT=$(command -v cut)
WC=$(command -v wc)
TR=$(command -v tr)
BC=$(command -v bc)

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
  export KMP_AFFINITY=compact,granularity=fine KMP_HW_SUBSET=1T
  export OMP_NUM_THREADS=$((NC/NS))
fi

if [ "" = "${LIBXSMM_TARGET_HIDDEN}" ] || [ "0" = "${LIBXSMM_TARGET_HIDDEN}" ]; then
  echo "OMP_NUM_THREADS=${OMP_NUM_THREADS} NUMACTL=\"${NUMACTL}\""
  echo
fi

if [ "${DATE}" ]; then
  LOGFILE=$(basename $0 .sh)-$(${DATE} +%Y%m%d-%H%M%S).log
else
  LOGFILE=$(basename $0 .sh).log
fi

EXE="${NUMACTL} ./layer_example_${TYPE}"

# arguments: iters ifw ifh nImg nIfm nOfm kw kh padw padh stride type format padmode
#
if [ "${BIN}" != "f32" ]; then
  true
else
${EXE} ${ITERS} 1920 1080 ${MB}   3  64 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee    ${LOGFILE}
fi
${EXE} ${ITERS} 1920 1080 ${MB}  64  64 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  960  540 ${MB}  64 128 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  960  540 ${MB} 128 128 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  480  270 ${MB} 128 256 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  480  270 ${MB} 256 256 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  480  270 ${MB} 256 256 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  240  135 ${MB} 256 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  240  135 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  240  135 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
${EXE} ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
RESULT=$?

if [ "0" = "${RESULT}" ] && [ "${GREP}" ] && [ "${PASTE}" ] && [ "${BC}" ]; then
  echo -n "GFLOPS: "
  echo "$(${GREP} "PERFDUMP" ${LOGFILE} | ${CUT} -d, -f16 | ${PASTE} -sd+ | ${BC})/13" | ${BC} -l
  echo -n "FPS:    "
  echo "${NS}/$(${GREP} "PERFDUMP" ${LOGFILE} | ${CUT} -d, -f15 | ${PASTE} -sd+ | ${BC})" | ${BC} -l
fi

exit ${RESULT}

