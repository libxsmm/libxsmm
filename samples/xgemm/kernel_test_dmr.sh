#!/usr/bin/env bash
# shellcheck disable=SC2012

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd "${HERE}" && ls -1 ./kernel_test/bf16*.slurm ./kernel_test/bf8*.slurm ./kernel_test/hf8*.slurm | grep -v spmm | ${EXEC} -c 3- "$@"
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
