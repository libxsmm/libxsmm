#!/usr/bin/env bash
# shellcheck disable=SC2012

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd "${HERE}" && ls -1 ./kernel_test/binary_*.sh ./kernel_test/unary_*.sh | ${EXEC} -c 3- "$@"
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
