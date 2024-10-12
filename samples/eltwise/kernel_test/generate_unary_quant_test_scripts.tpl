#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=10
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

PREC=PRECDESC

for LD in 'eqld' 'gtld'; do
  OUTNAME="${HERE}/unary_quant_"
  PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')

  OUTNAME=${OUTNAME}${PRECLC}_${LD}.sh

  # generate script by sed
  sed "s/PREC=0/PREC=\"${PREC}\"/g" ${HERE}/unary_quant.tpl \
  | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
  >${OUTNAME}

  # for gt we need to touch up the script
  if [ "$LD" == 'gtld' ] ; then
    sed "s/+ str(m) + '_' + str(m)/+ '100_100'/g" ${OUTNAME} >${TMPFILE}
    cp ${TMPFILE} ${OUTNAME}
  fi

  chmod 755 ${OUTNAME}
done
