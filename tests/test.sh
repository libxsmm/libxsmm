#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
GREP=$(which grep)
ENV=$(which env)

if [ "Windows_NT" = "${OS}" ]; then
  # Cygwin's ldd hangs with dyn. linked executables or certain shared libraries
  LDD=$(which cygcheck)
  # Cygwin's "env" does not set PATH ("Files/Black: No such file or directory")
  export PATH=${PATH}:${HERE}/../lib
else
  if [ "" != "$(which ldd)" ]; then
    LDD=ldd
  else
    LDD=echo
  fi
fi

${ECHO} "============="
${ECHO} "Running tests"
${ECHO} "============="
TESTS=$(ls -1 ${HERE}/*.c)
NTEST=1
NMAX=$(${ECHO} ${TESTS} | wc -w)
for TEST in ${TESTS} ; do
  NAME=$(basename ${TEST} .c)
  ${ECHO} -n "${NTEST} of ${NMAX} (${NAME})... "

  ERROR=$({
  if [ "-mic" != "$1" ]; then
    if [ "" != "$(${LDD} ${HERE}/${NAME} 2> /dev/null | ${GREP} libiomp5\.so)" ]; then
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../lib \
        KMP_AFFINITY=scatter,granularity=fine,1 \
        MIC_KMP_AFFINITY=scatter,granularity=fine \
        MIC_ENV_PREFIX=MIC \
        OFFLOAD_INIT=on_start \
      ${HERE}/${NAME} $*
    else
      ${ENV} LD_LIBRARY_PATH=${LD_LIBRARY_PATH}:${HERE}/../lib \
        DYLD_LIBRARY_PATH=${DYLD_LIBRARY_PATH}:${HERE}/../lib \
        OMP_PROC_BIND=TRUE \
      ${HERE}/${NAME} $*
    fi
  else
    shift
    ${ENV} \
      SINK_LD_LIBRARY_PATH=${SINK_LD_LIBRARY_PATH}:${MIC_LD_LIBRARY_PATH}:${HERE}/../lib/mic \
    micnativeloadex \
      ${HERE}/${NAME} -a "$*" \
      -e "KMP_AFFINITY=scatter,granularity=fine"
  fi > /dev/null; } 2>&1)
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} "FAILED(${RESULT}) ${ERROR}"
    exit ${RESULT}
  else
    ${ECHO} "OK ${ERROR}"
  fi
  NTEST=$((NTEST+1))
done

