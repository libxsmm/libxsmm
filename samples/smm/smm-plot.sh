#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
VARIANT=Cached

if [ "" != "$1" ]; then
  VARIANT=$1
  shift
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
  GNUPLOT_MAJOR=$("${GNUPLOT}" --version | sed "s/.\+ \([0-9]\).\([0-9]\) .*/\1/")
  GNUPLOT_MINOR=$("${GNUPLOT}" --version | sed "s/.\+ \([0-9]\).\([0-9]\) .*/\2/")
fi
GNUPLOT_VERSION=$((GNUPLOT_MAJOR * 10000 + GNUPLOT_MINOR * 100))

GREP=$(which grep)
SED=$(which sed)

function capturedTxtToDataFile {
  ${GREP} -i -A2 \
    -e "^m=" -e "${VARIANT}" \
    ${HERE}/$1.txt \
  | ${SED} \
    -e "s/m=//" -e "s/n=//" -e "s/k=//" -e "s/ (.\+) / /" \
    -e "s/size=//" -e "s/memory=//" -e "s/ GB\/s//" \
    -e "/^.\+\.\.\./Id" -e "/^$/d" -e "/--/d" \
  | ${SED} \
    -e "N;s/ MB\n\tperformance://g" \
    -e "N;s/ GFLOPS\/s\n\t.\+$//g" \
  > ${HERE}/$1.dat
}

if [ "40600" -le "${GNUPLOT_VERSION}" ]; then
  if [ "" = "$1" ]; then
    FILENAME=smm-$(echo ${VARIANT} | tr '[:upper:]' '[:lower:]').pdf
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

  capturedTxtToDataFile smm-blas
  capturedTxtToDataFile smm-dispatched
  capturedTxtToDataFile smm-inlined
  capturedTxtToDataFile smm-specialized

  env \
    GDFONTPATH=/cygdrive/c/Windows/Fonts \
    FILENAME=${FILENAME} \
    MULTI=${MULTI} \
  "${WGNUPLOT}" smm-perf.plt
fi

