#!/usr/bin/env bash
# shellcheck disable=SC2012

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd "${HERE}" && ls -1 ./kernel_test/f16f16f32*.slurm ./kernel_test/ace_k2_*f16f16f32*.slurm ./kernel_test/bf16*.slurm ./kernel_test/[su][su]i8*.slurm | grep -v spmm | ${EXEC} -c 3- "$@"
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
