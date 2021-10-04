#!/usr/bin/env bash
# shellcheck disable=SC2012

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

ls -1 "${HERE}"/kernel_test/*.slurm | ${EXEC}
rm -f tmp.??????????
