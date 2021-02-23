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
elif [ "opreduce" = "${KERNEL}" ]; then
  if [ $# = 11 ]; then
    M=$2
    N=$3
    N_COLS_IDX=$4
    LD=$5
    OP=$6
    OP_ORDER=$7
    SCALE_OP_RES=$8
    REDOP=$9
    ITERS=$10
    USE_BF16=$11
    ${NUMACTL} ./eltwise_opreduce ${M} ${N} ${N_COLS_IDX} ${LD} ${OP} ${OP_ORDER} ${SCALE_OP_RES} ${REDOP} ${ITERS} ${USE_BF16}
  else
    ITERS=100
    for M in 11 16 19 32 34 64 69 ; do
      for N in 27 32 45 64 ; do
        for (( N_COLS_IDX=0; N_COLS_IDX<=${N}; N_COLS_IDX+=10 )); do
          LD_LIST=( ${M} $(( M + 7 )) )
          for LD in "${LD_LIST[@]}" ; do
            for OP in 0 1 2 3 4; do
              for OP_ORDER in 0 1; do
                for SCALE_OP_RES in 0 1; do
                  for REDOP in 0 1 3 3; do
                    for USE_BF16 in 0 1; do
                      if [ ${OP} != 0 ] || [ ${REDOP} != 0 ]; then
                        ${NUMACTL} ./eltwise_opreduce ${M} ${N} ${N_COLS_IDX} ${LD} ${OP} ${OP_ORDER} ${SCALE_OP_RES} ${REDOP} ${ITERS} ${USE_BF16}
                      fi
                    done
                  done
                done
              done
            done
          done
        done
      done
    done
  fi
elif [ "scale" = "${KERNEL}" ]; then
  if [ $# = 11 ]; then
    M=$2
    N=$3
    LD_IN=$4
    LD_OUT=$5
    SHIFT=$6
    SCALE=$7
    BIAS=$8
    SCALE_ROWS=$9
    SCALE_ROWS_BCAST=$10
    ITERS=$11
    ${NUMACTL} ./eltwise_scale ${M} ${N} ${LD_IN} ${LD_OUT} ${SHIFT} ${SCALE} ${BIAS} ${SCALE_ROWS} ${SCALE_ROWS_BCAST} ${ITERS}
  else
    ITERS=100
    for M in 11 16 19 32 34 64 69 ; do
      for N in 27 32 45 64 ; do
        LD_LIST=( ${M} $(( M + 7 )) )
        for LD_IN in "${LD_LIST[@]}" ; do
          LD_OUT=${LD_IN}
          for SHIFT in 0 1; do
            for SCALE in 0 1; do
              for BIAS in 0 1; do
                for SCALE_ROWS in 0 1; do
                  if [ ${SHIFT} != 0 ] || [ ${SCALE} != 0 ] || [ ${BIAS} != 0 ]; then
                    ${NUMACTL} ./eltwise_scale ${M} ${N} ${LD_IN} ${LD_OUT} ${SHIFT} ${SCALE} ${BIAS} ${SCALE_ROWS} 0 ${ITERS}
                  else
                    if [ ${SCALE_ROWS} = 1 ] ; then
                      ${NUMACTL} ./eltwise_scale ${M} ${N} ${LD_IN} ${LD_OUT} ${SHIFT} ${SCALE} ${BIAS} ${SCALE_ROWS} 1 ${ITERS}
                    fi
                  fi
                done
              done
            done
          done
        done
      done
    done
  fi
elif [ "unary" = "${KERNEL}" ]; then
  if [[ $# == 11  || ( $# == 10 && $6 == 1 ) ]]; then
    M=$2
    N=$3
    LD_IN=$4
    LD_OUT=$5
    OP=$6
    PREC_IN=$7
    if [ ${OP} != 0 ] ; then
      PREC_COMP=$8
      PREC_OUT=$9
      BCAST_IN=$10
      ${NUMACTL} ./eltwise_unary ${OP} ${BCAST_IN} ${PREC_IN} ${PREC_COMP} ${PREC_OUT}  ${M} ${N} ${LD_IN} ${LD_OUT}
    else
      PREC_OUT=$8
      BITM=$9
      ${NUMACTL} ./eltwise_unary_relu ${OP} ${BITM} ${PREC_IN} ${PREC_OUT} ${M} ${N} ${LD_IN} ${LD_OUT}
    fi
  else
    for M in 11 16 19 32 34 64 69 2589; do
      for N in 27 32 45 64 712 ; do
        LD_LIST=( ${M} $(( M + 7 )) )
        for LD_IN in "${LD_LIST[@]}" ; do
          LD_OUT=${LD_IN}
          for OP in 0 2 3 4 5 6 7 8 9 10 11 12 13 14 15; do
            for PREC_IN in 2 4; do
              for PREC_COMP in 4; do
                for PREC_OUT in 2 4; do
                  for BCAST_IN in 0 1 2 3; do
                    ${NUMACTL} ./eltwise_unary ${OP} ${BCAST_IN} ${PREC_IN} ${PREC_COMP} ${PREC_OUT}  ${M} ${N} ${LD_IN} ${LD_OUT}
                  done
                done
              done
            done
          done
        done
      done
    done
    for M in 11 16 19 32 34 64 69 2589; do
      for N in 27 32 45 64 712 ; do
        LD_IN=${M}
        LD_OUT=${LD_IN}
        for OP in 'F' 'B'; do
          for PREC_IN in 2 4; do
            for PREC_OUT in 2 4; do
              for BITM in 0; do
                ${NUMACTL} ./eltwise_unary_relu ${OP} ${BITM} ${PREC_IN} ${PREC_OUT} ${M} ${N} ${LD_IN} ${LD_OUT}
              done
            done
          done
        done
      done
    done
  fi
fi


