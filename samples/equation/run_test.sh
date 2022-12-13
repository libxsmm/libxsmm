#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)
EXEC=${HERE}/../../scripts/tool_pexec.sh

cd "${HERE}" && ls -1 ./equation_test/*.sh | ${EXEC} -k 0 -c 3- "$@"
RESULT=$?

rm -f tmp.??????????
exit ${RESULT}
