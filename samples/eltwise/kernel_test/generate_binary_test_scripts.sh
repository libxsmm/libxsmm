#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'F32_F32_F32_F32' 'BF16_BF16_BF16_BF16' 'F32_F32_BF16_F32' 'F32_BF16_F32_F32' 'F32_BF16_BF16_F32' 'BF16_F32_F32_F32' 'BF16_F32_BF16_F32' 'BF16_BF16_F32_F32' 'BF16_BF16_BF16_F32' 'F16_F16_F16_F16' 'F32_F32_F16_F32' 'F32_F16_F32_F32' 'F32_F16_F16_F32' 'F16_F32_F32_F32' 'F16_F32_F16_F32' 'F16_F16_F32_F32' 'F16_F16_F16_F32' 'BF8_BF8_BF8_BF8' 'F32_F32_BF8_F32' 'F32_BF8_F32_F32' 'F32_BF8_BF8_F32' 'BF8_F32_F32_F32' 'BF8_F32_BF8_F32' 'BF8_BF8_F32_F32' 'BF8_BF8_BF8_F32' 'HF8_HF8_HF8_HF8' 'F32_F32_HF8_F32' 'F32_HF8_F32_F32' 'F32_HF8_HF8_F32' 'HF8_F32_F32_F32' 'HF8_F32_HF8_F32' 'HF8_HF8_F32_F32' 'HF8_HF8_HF8_F32' 'F64_F64_F64_F64'; do
  for TYPE in 1 2 3 4 5; do
    for LD in 'eqld' 'gtld'; do
      TPPNAME="none"
      OUTNAME="binary_"
      PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')

      # only cpy TPP has low precision compute
      if [[ (("$PREC" == 'F16_F16_F16_F16') || ("$PREC" == 'BF16_BF16_BF16_BF16') || ("$PREC" == 'BF8_BF8_BF8_BF8') || ("$PREC" == 'HF8_HF8_HF8_HF8')) ]]; then
        continue
      fi

      # get TPP name
      if [ "$TYPE" == '1' ] ; then
        TPPNAME="add"
      elif [ "$TYPE" == '2' ] ; then
        TPPNAME="mul"
      elif [ "$TYPE" == '3' ] ; then
        TPPNAME="sub"
      elif [ "$TYPE" == '4' ] ; then
        TPPNAME="div"
      elif [ "$TYPE" == '5' ] ; then
        TPPNAME="muladd"
      else
        continue
      fi

      OUTNAME=${OUTNAME}${TPPNAME}_${PRECLC}_${LD}.sh

      # generate script by sed
      sed "s/PREC=0/PREC=\"${PREC}\"/g" binary.tpl \
      | sed "s/BINARY_OP=0/BINARY_OP=${TYPE}/g" \
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
