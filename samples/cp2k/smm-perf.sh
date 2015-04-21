#!/bin/bash

VARIANT=Specialized

if ([ "" != "$1" ]) ; then
  VARIANT=$1
fi

HERE=$(cd $(dirname $0); pwd -P)
FILE=${HERE}/smm-test.txt

GREP=$(which grep)
PERF=$(${GREP} -A1 -i "${VARIANT}" ${FILE} | \
  ${GREP} -e "performance" | \
  cut -d" " -f2 | \
  sort -n)

NUM=$(echo "${PERF}" | wc -l)
MIN=$(echo ${PERF} | cut -d" " -f1)
MAX=$(echo ${PERF} | cut -d" " -f${NUM})

echo "num=${NUM}"
echo "min=${MIN}"
echo "max=${MAX}"

BC=$(which bc 2> /dev/null)
if ([ "" != "${BC}" ]) ; then
  AVG=$(echo "$(echo -n "scale=3;(${PERF})/${NUM}" | tr "\n" "+")" | ${BC})
  NUM2=$((NUM / 2))

  if ([ "0" == "$((NUM % 2))" ]) ; then
    A=$(echo ${PERF} | cut -d" " -f$((NUM2 - 1)))
    B=$(echo ${PERF} | cut -d" " -f${NUM2})
    MED=$(echo "$(echo -n "scale=3;(${A} + ${B})/2")" | ${BC})
  else
    MED=$(echo ${PERF} | cut -d" " -f${NUM2})
  fi

  echo "avg=${AVG}"
  echo "med=${MED}"
fi

if ([ -f /cygdrive/c/Program\ Files/gnuplot/bin/wgnuplot ]) ; then
  GNUPLOT=/cygdrive/c/Program\ Files/gnuplot/bin/wgnuplot
elif ([ -f /cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/wgnuplot ]) ; then
  GNUPLOT=/cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/wgnuplot
else
  GNUPLOT=$(which gnuplot 2> /dev/null)
fi

SED=$(which sed)
if ([ "" != "${GNUPLOT}" ]) ; then
  export GDFONTPATH=/cygdrive/c/Windows/Fonts
  ${GREP} -i -A1 -e "^m=" -e "${VARIANT}" ${FILE} | \
  ${SED} \
    -e "s/m=//" -e "s/n=//" -e "s/k=//" -e "s/ldc=//" -e "s/ (.\+) / /" \
    -e "s/size=//" -e "s/batch=//" -e "s/memory=//" -e "s/ GFLOPS\/s//" \
    -e "/${VARIANT}.../Id" -e "/^$/d" -e "/--/d" | \
  ${SED} \
    -e "N;s/ MB\n\tperformance://g" \
  > smm-test.dat
  "${GNUPLOT}" smm-test.plt
fi

