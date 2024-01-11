#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=10
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'F32_F32_IMPLICIT_F32_F32' 'BF16_BF16_IMPLICIT_BF16_F32' 'F16_F16_IMPLICIT_F16_F32' 'BF8_BF8_IMPLICIT_BF8_F32' 'HF8_HF8_IMPLICIT_HF8_F32'; do
  for TYPE in 1; do
    for ROUND in 'rne' 'stoch'; do
      for LD in 'eqld' 'gtld'; do
        TPPNAME="none"
        OUTNAME="${HERE}/ternary_"
        PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')
        RMODE=0

        # get TPP name
        if [ "$TYPE" == '1' ] ; then
          TPPNAME="select"
        else
          continue
        fi

        if [ "$ROUND" == 'stoch' ]; then
          PREC_OUT=$(echo "$PRECLC" |  awk -F"_" '{print $4}')
          if [ "$PREC_OUT" == 'bf8' ] ; then
            PREC_IN0=$(echo "$PRECLC" |  awk -F"_" '{print $1}')
            PREC_IN1=$(echo "$PRECLC" |  awk -F"_" '{print $2}')
            PREC_IN2=$(echo "$PRECLC" |  awk -F"_" '{print $3}')
            PREC_COMP=$(echo "$PRECLC" |  awk -F"_" '{print $5}')
            PREC_OUT=${PREC_OUT}${ROUND}
            RMODE=1
            PRECLC=${PREC_IN0}_${PREC_IN1}_${PREC_IN2}_${PREC_OUT}_${PREC_COMP}
          else
            continue
          fi
        fi

        OUTNAME=${OUTNAME}${TPPNAME}_${PRECLC}_${LD}.sh

        # generate script by sed
        sed "s/PREC=0/PREC=\"${PREC}\"/g" ${HERE}/ternary.tpl \
        | sed "s/TERNARY_OP=0/TERNARY_OP=${TYPE}/g" \
        | sed "s/RMODE=0/RMODE=${RMODE}/g" \
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
done
