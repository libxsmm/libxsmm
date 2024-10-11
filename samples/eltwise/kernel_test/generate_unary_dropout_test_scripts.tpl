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
  OUTNAME="${HERE}/unary_dropout_"
  PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')

  # only cpy TPP has low precision compute
  if [[ (("$PREC" == 'F16_F16_F16') || ("$PREC" == 'BF16_BF16_BF16') || ("$PREC" == 'BF8_BF8_BF8') || ("$PREC" == 'HF8_HF8_HF8') || ("$PREC" == 'F64_F64_F64')) ]]; then
    continue
  fi

  OUTNAME=${OUTNAME}${PRECLC}_${LD}.sh

  # generate script by sed
  sed "s/PREC=0/PREC=\"${PREC}\"/g" ${HERE}/unary_dropout.tpl \
  | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
  >${OUTNAME}

  # for gt we need to touch up the script
  if [ "$LD" == 'gtld' ] ; then
    sed "s/+ str(m) + '_' + str(m)/+ '100_100'/g" ${OUTNAME} >${TMPFILE}
    cp ${TMPFILE} ${OUTNAME}
  fi

  chmod 755 ${OUTNAME}
done
