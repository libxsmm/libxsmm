#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
NAME=$(basename $0)

TOUCH=$(which touch)
ECHO=$(which echo)
SED=$(which sed)
TR=$(which tr)

DEST=$1
if [ "$1" = "" ]; then
  DEST=.
fi

STATEFILE=${DEST}/.state
STATE=$(${TR} '?' '\n' | ${SED} -e 's/^  *//')

if [ ! -e ${STATEFILE} ] || [ 0 != $(${ECHO} "${STATE}" | diff -q ${STATEFILE} - > /dev/null; ${ECHO} $?) ]; then
  ${ECHO} "${STATE}" > ${STATEFILE}
  ${ECHO} $0
  ${TOUCH} $0
fi

