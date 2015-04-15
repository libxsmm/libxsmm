#!/bin/bash

VARIANT=Specialized

if ([ "" != "$1" ]) ; then
  VARIANT=$1
fi

PERF=$(grep -A1 -i "${VARIANT}" smm-test.txt | \
  grep -e "performance" | \
  cut -d" " -f2 | \
  sort -n)

#AVG=$((echo -n "scale=3;"; echo "${PERF}0" | tr "\n" "+") | bc)
#echo ${AVG}
NUM1=$(echo "${PERF}" | wc -l)
NUM2=$((NUM1 / 2))
MIN=$(echo ${PERF} | cut -d" " -f1)
MAX=$(echo ${PERF} | cut -d" " -f${NUM1})
AVG=$(echo "$(echo -n "scale=3;(${PERF})/${NUM1}" | tr "\n" "+")" | bc)

if ([ "0" == "$((NUM1 % 2))" ]) ; then
  A=$(echo ${PERF} | cut -d" " -f$((NUM2 - 1)))
  B=$(echo ${PERF} | cut -d" " -f${NUM2})
  MED=$(echo "$(echo -n "scale=3;(${A} + ${B})/2")" | bc)
else
  MED=$(echo ${PERF} | cut -d" " -f${NUM2})
fi

echo "num=${NUM1}"
echo "min=${MIN}"
echo "max=${MAX}"
echo "avg=${AVG}"
echo "med=${MED}"

