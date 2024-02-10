#!/usr/bin/env bash
# shellcheck disable=SC2012

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd "${HERE}" && ls -1 ./kernel_test/bf16*.slurm | grep -v spmm | ${EXEC} -o bf16only.out -c 3- "$@"
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
