#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
ECHO=$(which echo)
SCRT=${HERE}/../../scripts/libxsmm_utilities.py
FILE=cp2k-perf.txt

RUNS0=$(${SCRT} -1 $((64*64*64-0)) 19   23, 6, 14 16 29, 5 16 13 24 26, 9 16 22, 32, 64, 78, 16 29 55                                                         0 0)
RUNS1=$(${SCRT} -1 $((64*64*64-0)) 19   23, 6, 14 16 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55                                                         0 0)
RUNS2=$(${SCRT} -1 $((64*64*64-0)) 20   23, 6, 14 16 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 12                                                     0 0)
RUNS3=$(${SCRT} -1 $((64*64*64-0)) 26   23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12                                 0 0)
RUNS4=$(${SCRT} -1 $((64*64*64-1)) 31   23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45                 0 0)
RUNS5=$(${SCRT} -1 $((64*64*64-0)) 31   23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45                 0 0)
RUNS6=$(${SCRT} -1 $((80*80*80-0)) 35   23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 32, 64, 78, 16 29 55, 32 29 55, 12, 13 26 28 32 45, 7 13 25 32     0 0)
RUNS7=$(${SCRT} -1 $((80*80*80-0)) 35   23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45          0 0)
RUNS8=$(${SCRT} -1 $((80*80*80-0)) 35   23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10    0 0)
RUNS9=$(${SCRT} -1 $((80*80*80-0)) 35   23, 6, 14 16 29, 14 32 29, 5 32 13 24 26, 9 32 22, 64, 78, 16 29 55, 32 29 55, 12, 4 5 7 9 13 25 26 28 32 45, 4 10 15 0 0)

case "$1" in
  "-"*) RUNS=RUNS${1:1}; shift
  ;;
esac

if [ "" = "${!RUNS}" ]; then
  RUNS=RUNS9
fi

if [ "" != "$1" ]; then
  FILE=$1
  shift
fi
if [ "" != "$1" ]; then
  SIZE=$1
  shift
else
  SIZE=0
fi
cat /dev/null > ${FILE}

NRUN=1
NMAX=$(${ECHO} ${!RUNS} | wc -w)
for RUN in ${!RUNS} ; do
  MVALUE=$(${ECHO} ${RUN} | cut --output-delimiter=' ' -d_ -f1)
  NVALUE=$(${ECHO} ${RUN} | cut --output-delimiter=' ' -d_ -f2)
  KVALUE=$(${ECHO} ${RUN} | cut --output-delimiter=' ' -d_ -f3)
  >&2 ${ECHO} -n "${NRUN} of ${NMAX} (M=${MVALUE} N=${NVALUE} K=${KVALUE})... "
  ERROR=$({ CHECK=1 ${HERE}/cp2k.sh ${MVALUE} ${SIZE} 0 ${NVALUE} ${KVALUE} >> ${FILE}; } 2>&1)
  RESULT=$?
  if [ 0 != ${RESULT} ]; then
    ${ECHO} "FAILED(${RESULT}) ${ERROR}"
    exit 1
  else
    ${ECHO} "OK ${ERROR}"
  fi
  ${ECHO} >> ${FILE}
  NRUN=$((NRUN+1))
done

