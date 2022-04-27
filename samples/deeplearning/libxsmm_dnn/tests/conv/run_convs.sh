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

if [ $# -ne 10 ]
then
  echo "Usage: $(basename $0) mb iters numa (1-mcdram/0-DDR) TYPE ('A'-ALL/'F'-FP/'B'-BP/'U'-WU) Padding Fuse(0 nothing, 1 Bias, 2 ReLU, 3 ReLU+Bias) BC BK TOPO (0 All, 1 Resnet 1.5, 2 Alexnet, 3 dcGAN, 4 deepbench, 5 GooglenetV1, 6 GooglenetV3, 7 maskrcnn, 8 overfeat, 9 VGGA, 10 VGGA high-res), using default values: 64 1000 1 f32 A 1 0 64 64 1"
  MB=${CHECK_DNN_MB}
  ITERS=${CHECK_DNN_ITERS}
  NUMA=-1
  BIN=f32
  TYPE="A"
  PAD=1
  FUSE=0
  BC=64
  BK=64
  TOPO=1
else
  MB=$1
  ITERS=$2
  NUMA=$3
  BIN=$4
  TYPE=$5
  PAD=$6
  FUSE=$7
  BC=$8
  BK=$9
  TOPO=${10}
fi

if [ "${UNAME}" ] && [ "${CUT}" ] && [ "x86_64" = "$(${UNAME} -m)" ]; then
  if [ "${GREP}" ] && [ "${CUT}" ] && [ "${SORT}" ] && [ "${WC}" ] && [ -e /proc/cpuinfo ]; then
    export NS=$(${GREP} "physical id" /proc/cpuinfo | ${SORT} -u | ${WC} -l | ${TR} -d " ")
    export NC=$((NS*$(${GREP} -m1 "cpu cores" /proc/cpuinfo | ${TR} -d " " | ${CUT} -d: -f2)))
    export NT=$(${GREP} "core id" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
  elif [ "${UNAME}" ] && [ "${CUT}" ] && [ "Darwin" = "$(${UNAME})" ]; then
    export NS=$(sysctl hw.packages | ${CUT} -d: -f2 | tr -d " ")
    export NC=$(sysctl hw.physicalcpu | ${CUT} -d: -f2 | tr -d " ")
    export NT=$(sysctl hw.logicalcpu | ${CUT} -d: -f2 | tr -d " ")
  fi
elif [ "${UNAME}" ] && [ "${CUT}" ] && [ "aarch64" = "$(${UNAME} -m)" ]; then
  export NS=1
  export NC=$(${GREP} "Features" /proc/cpuinfo | ${WC} -l | ${TR} -d " ")
  export NT=$NC
fi
if [ "${NC}" ] && [ "${NT}" ]; then
  export HT=$((NT/(NC)))
else
  export NS=1 NC=1 NT=1 HT=1
fi
if [ "${GREP}" ] && [ "${CUT}" ] && [ "$(command -v numactl)" ]; then
  export NN=$(numactl -H | ${GREP} "available:" | ${CUT} -d' ' -f2)
else
  export NN=${NS}
fi

if [ "${UNAME}" ] && [ "${CUT}" ] && [ "x86_64" = "$(${UNAME} -m)" ]; then
  CPUFLAGS=$(if [ "${GREP}" ] && [ "${CUT}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | ${CUT} -d: -f2- || true; fi)
else
  CPUFLAGS=
fi
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

# ./layer_example iters inpWidth inpHeight nImg nIfm nOfm kw kh padw padh stride type
#
if [ "f32" == "${BIN}" ]; then
  PREC_BF16=0
else
  PREC_BF16=1
fi

if [ ${TOPO} -eq 1 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS}  224 224 ${MB}     3   64 7 7 3 3 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS}  224 224 ${MB}     4   64 7 7 3 3 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS}   56  56 ${MB}   128  128 3 3 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   56  56 ${MB}    64   64 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   56  56 ${MB}   256  512 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   56  56 ${MB}    64   64 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   56  56 ${MB}    64  256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   56  56 ${MB}   256   64 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   56  56 ${MB}   256  128 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   28  28 ${MB}   256  256 3 3 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   28  28 ${MB}   128  128 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   28  28 ${MB}   512 1024 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   28  28 ${MB}   512  256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   28  28 ${MB}   512  128 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   28  28 ${MB}   128  512 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   14  14 ${MB}   512  512 3 3 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   14  14 ${MB}   256  256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   14  14 ${MB}  1024 2048 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   14  14 ${MB}   256 1024 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   14  14 ${MB}  1024  512 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   14  14 ${MB}  1024  256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    7   7 ${MB}   512  512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    7   7 ${MB}   512 2048 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    7   7 ${MB}  2048  512 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 2 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS} 227 227  ${MB}    3   64 11 11 0 0 4 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS} 227 227  ${MB}    4   64 11 11 0 0 4 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS}  27  27  ${MB}   64  192  5  5 2 2 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  13  13  ${MB}  192  384  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  13  13  ${MB}  384  256  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  13  13  ${MB}  256  256  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 3 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS}  64  64  ${MB}    3   64  4  4 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS}  64  64  ${MB}    4   64  4  4 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS}   4   4  ${MB}  512  100  4  4 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   4   4  ${MB}  512   96  4  4 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   8   8  ${MB}  256  512  4  4 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  16  16  ${MB}  128  256  4  4 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  32  32  ${MB}   64  128  4  4 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}   4   4  ${MB}  512  512  4  4 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 4 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL}  ./layer_example  ${ITERS}  224 224 16     3   64   7 7 3 3 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  224 224 16     3   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  108 108  8     3   64   3 3 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  224 224  8     3   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161  4     1   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161  8     1   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161 16     1   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161 32     1   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  480  48 16     1   16   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161 16     1   64   5 5 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL}  ./layer_example  ${ITERS}  224 224 16     4   64   7 7 3 3 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  224 224 16     4   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  108 108  8     4   64   3 3 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  224 224  8     4   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161  4     2   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161  8     2   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161 16     2   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161 32     2   32  20 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  480  48 16     2   16   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  700 161 16     2   64   5 5 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL}  ./layer_example  ${ITERS}  341  79  4    32   32  10 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  341  79  8    32   32  10 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  341  79 16    32   32  10 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  341  79 32    32   32  10 5 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  240  24 16    16   32   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  120  12 16    32   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   60   6 16    64  128   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   54  54  8    64   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   27  27  8   128  128   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8   128  256   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7  8   256  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  112 112  8    64  128   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56  8   128  256   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28  8   256  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8   512  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7  8   512  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  112 112 16    64  128   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56 16   128  256   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   256  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   512  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16   512  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   192   32   5 5 2 2 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   192   64   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   512   48   5 5 2 2 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   512  192   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16   832  256   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16   832  128   5 5 2 2 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56  8    64   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56  8    64  256   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28  8   128  128   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28  8   128  512   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8   256  256   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8   256  256   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8   256 1024   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7  8   512  512   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7  8  2048  512   1 1 3 3 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56 16    64   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56 16    64  256   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   128  128   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   128  512   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   256  256   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   256  256   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   256 1024   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16   512  512   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16  2048  512   1 1 3 3 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  350  80 16    64   64   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  350  80 16    64  128   5 5 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  175  40 16   128  128   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  175  40 16   128  256   5 5 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   84  20 16   256  256   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   84  20 16   256  512   5 5 1 1 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   42  10 16   512  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  112 112  8    64   64   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56  8    64  256   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56  8   256   64   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56  8   256  128   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28  8   128  512   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28  8   512  128   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28  8   512  256   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8   256 1024   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28  8   512 1024   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8  1024  256   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8   256 1024   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8  1024  512   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7  8   512  512   3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7  8   512 2048   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14  8  1024 2048   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7  8  2048  512   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}  112 112 16    64   64   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56 16    64  256   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56 16   256   64   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   56  56 16   256  128   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   128  512   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   512  128   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   512  256   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   256 1024   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   28  28 16   512 1024   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16  1024  256   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16   256 1024   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16  1024  512   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16   512 512    3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16   512 2048   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}   14  14 16  1024 2048   1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL}  ./layer_example  ${ITERS}    7   7 16  2048  512   1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 5 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS}   224  224  ${MB}     3    64  7  7  3  3  2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS}   224  224  ${MB}     4    64  7  7  3  3  2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS}    56   56  ${MB}    64    64  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    56   56  ${MB}    64   192  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   192    64  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   192    96  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}    96   128  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   192    16  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}    16    32  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   192    32  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   256   128  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   128   192  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   256    32  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}    32    96  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    28   28  ${MB}   256    64  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   480   192  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   480    96  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}    96   208  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   480    16  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}    16    48  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   480    64  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512   160  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512   112  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   112   224  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512    32  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}    32    64  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512    64  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512   128  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   128   256  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512   144  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   144   288  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512    32  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}    32    64  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   528   256  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   528   160  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   160   320  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   528    32  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}    32   128  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   528   128  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   832   256  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   832   160  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   160   320  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   832    32  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}    32   128  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   832   128  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   832   384  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   832   192  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   192   384  3  3  1  1  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}   832    48  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}     7    7  ${MB}    48   128  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
#${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}   512    24  1  1  0  0  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
#${NUMACTL} ./layer_example ${ITERS}    14   14  ${MB}    24    64  5  5  2  2  1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 6 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS}  299 299 ${MB}     3   32 3 3 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS}  299 299 ${MB}     4   32 3 3 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS}  149 149 ${MB}    32   32 3 3 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  147 147 ${MB}    32   64 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  73  73  ${MB}    64   80 3 3 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  73  73  ${MB}    80  192 3 3 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  71  71  ${MB}    80  192 3 3 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   192   64 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}    64   96 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}    96   96 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   192   48 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}    48   64 5 5 2 2 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   192   32 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   256   64 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   256   48 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   288   64 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   288   48 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}    96   96 3 3 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  35  35  ${MB}   288  384 3 3 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   768  128 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   128  128 1 7 0 3 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   128  128 7 1 3 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   128  192 7 1 3 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   128  192 1 7 0 3 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   768  192 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   768  160 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   160  160 1 7 0 3 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   160  160 7 1 3 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   160  192 7 1 3 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   160  192 1 7 0 3 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   192  192 1 7 0 3 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   192  192 7 1 3 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   192  192 3 3 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  17  17  ${MB}   192  320 3 3 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  1280  320 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  1280  192 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  1280  448 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}   448  384 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}   384  384 1 3 0 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}   384  384 3 1 1 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  1280  384 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  2048  320 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  2048  192 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  2048  448 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}  2048  384 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  8   8   ${MB}   384  384 1 3 0 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 7 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS} 1030 1030 ${MB} 3 64 7 7 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS} 1030 1030 ${MB} 4 64 7 7 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 64 256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 64 64 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 64 64 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 256 64 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 256 128 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 256 256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 256 512 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 128 128 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 128 512 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 512 128 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 512 256 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 64 64 ${MB} 256 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 64 64 ${MB} 256 1024 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 512 256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 512 1024 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 64 64 ${MB} 1024 256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 64 64 ${MB} 1024 2048 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 64 64 ${MB} 1024 512 1 1 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 32 32 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 32 32 ${MB} 512 2048 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 32 32 ${MB} 2048 512 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 32 32 ${MB} 2048 256 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 32 32 ${MB} 256 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 32 32 ${MB} 256 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 16 16 ${MB} 256 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 64 64 ${MB} 256 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 256 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 128 128 ${MB} 256 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 256 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 256 256 ${MB} 256 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 7 7 ${MB} 256 1024 7 7 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 1 1 ${MB} 1024 1024 1 1 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 14 14 ${MB} 256 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 28 28 ${MB} 256 256 2 2 0 0 2 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 8 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS} 231 231  ${MB}    3   96 11 11 0 0 4 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS} 231 231  ${MB}    4  96 11 11 0 0 4 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS}  28  28  ${MB}   96  256  5  5 0 0 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  12  12  ${MB}  256  512  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  12  12  ${MB}  512 1024  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  12  12  ${MB} 1024 1024  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 9 ] || [ ${TOPO} -eq 0 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS} 224 224  ${MB}    3   64  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS} 224 224  ${MB}    4   64  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS} 112 112  ${MB}   64  128  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  56  56  ${MB}  128  256  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  56  56  ${MB}  256  256  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  28  28  ${MB}  256  512  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  28  28  ${MB}  512  512  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  14  14  ${MB}  512  512  3  3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

if [ ${TOPO} -eq 10 ]; then
if [ ${PREC_BF16} -eq 0 ]; then
${NUMACTL} ./layer_example ${ITERS} 3840 2160 ${MB}   3  64 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 1920 1080 ${MB}   3  64 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
else
${NUMACTL} ./layer_example ${ITERS} 3840 2160 ${MB}   4  64 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 1920 1080 ${MB}   4  64 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi
${NUMACTL} ./layer_example ${ITERS} 1920 1080 ${MB}  64 128 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS} 1920 1080 ${MB} 128 128 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  960  540 ${MB} 128 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  960  540 ${MB} 256 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  960  540 ${MB} 256 256 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  480  270 ${MB} 256 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  480  270 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  480  270 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  240  135 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  240  135 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  240  135 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  960  540 ${MB}  64 128 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  960  540 ${MB} 128 128 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
${NUMACTL} ./layer_example ${ITERS}  120   68 ${MB} 512 512 3 3 1 1 1 ${TYPE} L ${PAD} ${FUSE} ${BC} ${BK} ${PREC_BF16}
fi

