#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'I8' 'I16' 'I32' 'I64' 'BF8' 'HF8' 'BF16' 'F16' 'F32' 'F64'; do
  for TYPE in 0 1; do
    for LD in 'eqld' 'gtld'; do
      TPPNAME="none"
      OUTNAME="unary_"
      NUMPREC=10
      PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')

      # only 16 and 32bit support
      if [[ (("$PREC" == 'I8') || ("$PREC" == 'BF8') || ("$PREC" == 'HF8') || ("$PREC" == 'F64') || ("$PREC" == 'I64')) ]]; then
        continue
      fi

      # get TPP name
      if [ "$TYPE" == '0' ] ; then
        TPPNAME="gather"
      elif [ "$TYPE" == '1' ] ; then
        TPPNAME="scatter"
      else
        continue
      fi

      if [[ (("$PREC" == 'I32') || ("$PREC" == 'F32')) ]]; then
        NUMPREC=0
      fi

      if [[ (("$PREC" == 'I16') || ("$PREC" == 'F16') || ("$PREC" == 'BF16')) ]]; then
        NUMPREC=1
      fi

      OUTNAME=${OUTNAME}${TPPNAME}_${PRECLC}_${LD}.sh

      # generate script by sed
      sed "s/NUMPREC=X/NUMPREC=\"${NUMPREC}\"/g" unary_gather_scatter.tpl \
      | sed "s/GS_OP=X/GS_OP=${TYPE}/g" \
      | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
      >${OUTNAME}

      # for gt we need to touch up the script
      if [ "$LD" == 'gtld' ] ; then
        sed "s/+ str(m) + '_' + str(m)/+ '100_100'/g" ${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${OUTNAME}
      fi

      chmod 755 ${OUTNAME}
    done
  done
done
