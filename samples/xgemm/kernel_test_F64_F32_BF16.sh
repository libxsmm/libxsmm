#!/usr/bin/env bash
# shellcheck disable=SC2012

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd "${HERE}" && \
ls -1 ./kernel_test/*.slurm \
  | grep -v -e "flat" -e "bf8" -e "i16" -e "sui8" -e "usi8" \
  | ${EXEC} -c 3- "$@"
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
