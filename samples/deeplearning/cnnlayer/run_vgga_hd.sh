#!/bin/bash

if [ "" = "${CHECK}" ] || [ "0" = "${CHECK}" ]; then
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=100; fi
else # check
  if [ "" = "${CHECK_DNN_ITERS}" ]; then CHECK_DNN_ITERS=1; fi
fi

if [ $# -ne 7 ]; then
  if [ $# -ne 0 ]; then
    echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) FORMAT ('A'-ALL/'L'-LIBXSMM/'T'-Tensorflow/'M'-Mixed) padding; using default values; using default values: 128 1000 1 f32 A L 0"
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

UNAME=$(which uname 2>/dev/null)
PASTE=$(which paste 2>/dev/null)
GREP=$(which grep 2>/dev/null)
SORT=$(which sort 2>/dev/null)
DATE=$(which date 2>/dev/null)
CUT=$(which cut 2>/dev/null)
WC=$(which wc 2>/dev/null)
TR=$(which tr 2>/dev/null)
BC=$(which bc 2>/dev/null)

if [ "" != "${GREP}" ] && [ "" != "${SORT}" ] && [ "" != "${WC}" ] && [ -e /proc/cpuinfo ]; then
  export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)
  export NC=$((NS*$(${GREP} "core id" /proc/cpuinfo | ${SORT} -u | ${WC} -l)))
  export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l)
elif [ "" != "${UNAME}" ] && [ "" != "${CUT}" ] && [ "Darwin" = "$(${UNAME})" ]; then
  export NS=$(sysctl hw.packages | ${CUT} -d: -f2 | tr -d " ")
  export NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | tr -d " ")
  export NT=$(sysctl hw.logicalcpu | ${CUT} -d: -f2 | tr -d " ")
fi
if [ "" != "${NC}" ] && [ "" != "${NT}" ]; then
  export HT=$((NT/(NC)))
else
  export NS=1 NC=1 NT=1 HT=1
fi
if [ "" != "${CUT}" ] && [ "" != "$(which numactl 2>/dev/null)" ]; then
  export NN=$(numactl -H | ${GREP} available: | ${CUT} -d' ' -f2)
else
  export NN=${NS}
fi

CPUFLAGS=$(if [ "" != "${GREP}" ] && [ "" != "${CUT}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | ${CUT} -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | ${GREP} -o avx512er)" ]; then
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

if [ "" != "${DATE}" ]; then
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
fi && \
${EXE} ${ITERS} 1920 1080 ${MB}  64  64 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  960  540 ${MB}  64 128 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  960  540 ${MB} 128 128 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  480  270 ${MB} 128 256 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  480  270 ${MB} 256 256 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  480  270 ${MB} 256 256 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  240  135 ${MB} 256 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  240  135 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  240  135 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}  && \
${EXE} ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${KIND} ${FORMAT} ${PADMODE} | tee -a ${LOGFILE}
RESULT=$?

if [ "0" = "${RESULT}" ] && [ "" != "${GREP}" ] && [ "" != "${PASTE}" ] && [ "" != "${BC}" ]; then
  echo -n "GFLOPS: "
  echo "$(${GREP} "PERFDUMP" ${LOGFILE} | ${CUT} -d, -f16 | ${PASTE} -sd+ | ${BC})/13" | ${BC} -l
  echo -n "FPS:    "
  echo "${NS}/$(${GREP} "PERFDUMP" ${LOGFILE} | ${CUT} -d, -f15 | ${PASTE} -sd+ | ${BC})" | ${BC} -l
fi

exit ${RESULT}

