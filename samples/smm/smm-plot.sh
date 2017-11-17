#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
VARIANT=Cached
LIMIT=31

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

SED=$(which sed)
function capturedTxtToDataFile {
  ${SED} \
    -e "/^m=/,/${VARIANT}/{//!d}" \
    -e "/${VARIANT}/d" \
    -e "/\.\.\./,/Finished/{//!d}" \
    -e "/Finished/d" \
    -e "/diff:/d" \
    -e "/\.\.\./d" \
    -e "/^$/d" \
    ${HERE}/$1.txt \
  | ${SED} \
    -e "s/m=//" -e "s/n=//" -e "s/k=//" -e "s/ (..*) / /" \
    -e "s/size=//" \
    -e "/duration:/d" \
  | ${SED} \
    -e "N;s/ memory=..*\n..*//" \
    -e "N;s/\n\tperformance:\(..*\) GFLOPS\/s/\1/" \
    -e "N;s/\n\tbandwidth:\(..*\) GB\/s/\1/" \
  > ${HERE}/$1.dat
}

if [ "40600" -le "${GNUPLOT_VERSION}" ]; then
  RM=$(which rm)
  if [ "" = "$1" ]; then
    FILENAME=smm-$(echo ${VARIANT} | tr ' ,' '-' | tr -d '()' | tr '[:upper:]' '[:lower:]').pdf
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

  ${RM} -f *.dat
  capturedTxtToDataFile smm-blas
  capturedTxtToDataFile smm-specialized
  #capturedTxtToDataFile smm-dispatched
  #capturedTxtToDataFile smm-inlined

  env \
    GDFONTPATH=/cygdrive/c/Windows/Fonts \
    FILENAME=${FILENAME} \
    MULTI=${MULTI} \
    LIMIT=${LIMIT} \
  "${WGNUPLOT}" smm-perf.plt
fi

