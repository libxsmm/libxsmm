#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

export LIBXSMM_TARGET=hsw
cd ${HERE} && cat <<EOM | ${EXEC} -c 3- "$@"
./equation_test/equation_simple.sh
./equation_test/equation_relu.sh
./equation_test/equation_gather_reduce.sh
./equation_test/equation_softmax.sh
./equation_test/equation_layernorm.sh
EOM
RESULT=$?

rm -f tmp.??????????
unset LIBXSMM_TARGET
exit ${RESULT}
