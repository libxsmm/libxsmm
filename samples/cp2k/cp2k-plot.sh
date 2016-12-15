#!/bin/sh

SORT=$(which sort)
SED=$(which sed)
CUT=$(which cut)

VARIANT=Specialized

if [ "" != "$1" ]; then
  VARIANT=$1
  shift
fi

HERE=$(cd $(dirname $0); pwd -P)
FILE=${HERE}/cp2k-perf.txt

GREP=$(which grep)
PERF=$(${GREP} -A1 -i "${VARIANT}" ${FILE} | \
  ${GREP} -e "performance" | \
  ${CUT} -d" " -f2 | \
  ${SORT} -n)

ECHO=$(which echo)
NUM=$(${ECHO} "${PERF}" | wc -l)
MIN=$(${ECHO} ${PERF} | ${CUT} -d" " -f1)
MAX=$(${ECHO} ${PERF} | ${CUT} -d" " -f${NUM})

${ECHO} "num=${NUM}"
${ECHO} "min=${MIN}"
${ECHO} "max=${MAX}"

BC=$(which bc 2> /dev/null)
if [ "" != "${BC}" ]; then
  AVG=$(${ECHO} "$(${ECHO} -n "scale=3;(${PERF})/${NUM}" | tr "\n" "+")" | ${BC})
  NUM2=$((NUM / 2))

  if [ "0" = "$((NUM % 2))" ]; then
    A=$(${ECHO} ${PERF} | ${CUT} -d" " -f${NUM2})
    B=$(${ECHO} ${PERF} | ${CUT} -d" " -f$((NUM2 + 1)))
    MED=$(${ECHO} "$(${ECHO} -n "scale=3;(${A} + ${B})/2")" | ${BC})
  else
    MED=$(${ECHO} ${PERF} | ${CUT} -d" " -f$((NUM2 + 1)))
  fi

  ${ECHO} "avg=${AVG}"
  ${ECHO} "med=${MED}"
fi

if [ -f /cygdrive/c/Program\ Files/gnuplot/bin/wgnuplot ]; then
  WGNUPLOT=/cygdrive/c/Program\ Files/gnuplot/bin/wgnuplot
  GNUPLOT=/cygdrive/c/Program\ Files/gnuplot/bin/gnuplot
elif [ -f /cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/wgnuplot ]; then
  WGNUPLOT=/cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/wgnuplot
  GNUPLOT=/cygdrive/c/Program\ Files\ \(x86\)/gnuplot/bin/gnuplot
else
  GNUPLOT=$(which gnuplot 2> /dev/null)
  WGNUPLOT=${GNUPLOT}
fi

GNUPLOT_MAJOR=0
GNUPLOT_MINOR=0
if [ -f "${GNUPLOT}" ]; then
  GNUPLOT_MAJOR=$("${GNUPLOT}" --version | ${SED} "s/.\+ \([0-9]\).\([0-9]\) .*/\1/")
  GNUPLOT_MINOR=$("${GNUPLOT}" --version | ${SED} "s/.\+ \([0-9]\).\([0-9]\) .*/\2/")
fi
GNUPLOT_VERSION=$((GNUPLOT_MAJOR * 10000 + GNUPLOT_MINOR * 100))

if [ "40600" -le "${GNUPLOT_VERSION}" ]; then
  if [ "" = "$1" ]; then
    FILENAME=cp2k-$(${ECHO} ${VARIANT} | tr '[:upper:]' '[:lower:]').pdf
  else
    FILENAME=$1
    shift
  fi
  if [ "" = "$1" ]; then
    MULTI=1
  else
    MULTI=$1
    shift
  fi
  ${GREP} -i -A2 \
    -e "^m=" -e "${VARIANT}" \
    ${FILE} | \
  ${SED} \
    -e "s/m=//" -e "s/n=//" -e "s/k=//" -e "s/ldc=[0-9][0-9]* //" -e "s/ (..*) / /" \
    -e "s/size=//" -e "s/batch=[0-9][0-9]* //" -e "s/memory=//" -e "s/ GB\/s//" \
    -e "/^..*\.\.\./Id" -e "/^$/d" -e "/--/d" | \
  ${SED} \
    -e "N;s/ MB\( (.P)\)*\n\tperformance://g" \
    -e "N;s/ GFLOPS\/s\n\tbandwidth://g" \
  > ${HERE}/cp2k-perf.dat
  env \
    GDFONTPATH=/cygdrive/c/Windows/Fonts \
    FILENAME=${FILENAME} \
    MULTI=${MULTI} \
  "${WGNUPLOT}" cp2k-perf.plt
fi

