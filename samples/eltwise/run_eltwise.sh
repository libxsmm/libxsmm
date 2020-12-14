#!/usr/bin/env bash
set -eo pipefail

UNAME=$(command -v uname)
SORT=$(command -v sort)
GREP=$(command -v grep)
CUT=$(command -v cut)
WC=$(command -v wc)
TR=$(command -v tr)
NUMA=-1

KERNEL=$1

if [ "" != "${GREP}" ] && [ "" != "${CUT}" ] && [ "" != "${SORT}" ] && [ "" != "${WC}" ] && [ -e /proc/cpuinfo ]; then
  export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
  export NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
  export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
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
if [ "" != "${GREP}" ] && [ "" != "${CUT}" ] && [ "" != "$(command -v numactl)" ]; then
  export NN=$(numactl -H | ${GREP} available: | ${CUT} -d' ' -f2)
else
  export NN=${NS}
fi

CPUFLAGS=$(if [ "" != "${GREP}" ] && [ "" != "${CUT}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | ${CUT} -d: -f2-; fi)
if [ "" != "${GREP}" ] && [ "" != "$(echo "${CPUFLAGS}" | ${GREP} -o avx512er)" ]; then
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

if [ "reduce" = "${KERNEL}" ]; then
  if [ $# = 8 ] || [ $# = 9 ]; then
    M=$2
    N=$3
    LD=$4
    REDUCE_ELTS=$5 # 0 or 1
    REDUCE_ELTS_SQUARED=$6 # 0 or 1
    REDUCE_ROWS=$7 # 0 for columns, 1 for rows
    ITERS=$8
    N_COLS_IDX=$9 # if set, this is the number of columns to be reduced
    ${NUMACTL} ./eltwise_reduce ${M} ${N} ${LD} ${REDUCE_ELTS} ${REDUCE_ELTS_SQUARED} ${REDUCE_ROWS} ${ITERS} ${N_COLS_IDX}
  else
    ITERS=100
    for M in 11 16 19 32 34 64 69 ; do
      for N in 27 32 45 64 ; do
        LD_LIST=( ${M} $(( M + 7 )) )
        for LD in "${LD_LIST[@]}" ; do
          for REDUCE_ELTS in 0 1; do
            for REDUCE_ELTS_SQUARED in 0 1; do
              for REDUCE_ROWS in 0 1; do
                if [ ${REDUCE_ELTS} != 0 ] ||  [ ${REDUCE_ELTS_SQUARED} != 0 ]; then
                  if [ ${REDUCE_ROWS} = 1  ]; then
                    ${NUMACTL} ./eltwise_reduce ${M} ${N} ${LD} ${REDUCE_ELTS} ${REDUCE_ELTS_SQUARED} ${REDUCE_ROWS} ${ITERS}
                  else
                    for (( N_COLS_IDX=0; N_COLS_IDX<=${N}; N_COLS_IDX+=10 )); do
                      ${NUMACTL} ./eltwise_reduce ${M} ${N} ${LD} ${REDUCE_ELTS} ${REDUCE_ELTS_SQUARED} ${REDUCE_ROWS} ${ITERS} ${N_COLS_IDX}
                    done
                  fi
                fi
              done
            done
          done
        done
      done
    done
  fi
fi


