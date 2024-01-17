#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=10
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'F32_F32_F32_F32' 'BF16_BF16_BF16_BF16' 'F32_F32_BF16_F32' 'F32_BF16_F32_F32' 'F32_BF16_BF16_F32' 'BF16_F32_F32_F32' 'BF16_F32_BF16_F32' 'BF16_BF16_F32_F32' 'BF16_BF16_BF16_F32' 'F16_F16_F16_F16' 'F32_F32_F16_F32' 'F32_F16_F32_F32' 'F32_F16_F16_F32' 'F16_F32_F32_F32' 'F16_F32_F16_F32' 'F16_F16_F32_F32' 'F16_F16_F16_F32' 'BF8_BF8_BF8_BF8' 'F32_F32_BF8_F32' 'F32_BF8_F32_F32' 'F32_BF8_BF8_F32' 'BF8_F32_F32_F32' 'BF8_F32_BF8_F32' 'BF8_BF8_F32_F32' 'BF8_BF8_BF8_F32' 'HF8_HF8_HF8_HF8' 'F32_F32_HF8_F32' 'F32_HF8_F32_F32' 'F32_HF8_HF8_F32' 'HF8_F32_F32_F32' 'HF8_F32_HF8_F32' 'HF8_HF8_F32_F32' 'HF8_HF8_HF8_F32' 'F64_F64_F64_F64' 'U16_U16_U32_IMPLICIT'; do
  for TYPE in 1 2 3 4 5 6 9 10 27 28 29 30 31 32; do
    for ROUND in 'rne' 'stoch'; do
      for LD in 'eqld' 'gtld'; do
        TPPNAME="none"
        OUTNAME="${HERE}/binary_"
        PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')
        RMODE=0
        PRECSCRIPT=${PREC}

        # only cpy TPP has low precision compute
        if [[ (("$PREC" == 'F16_F16_F16_F16') || ("$PREC" == 'BF16_BF16_BF16_BF16') || ("$PREC" == 'BF8_BF8_BF8_BF8') || ("$PREC" == 'HF8_HF8_HF8_HF8')) ]]; then
          continue
        fi

        # Binary zip tp blocks 2 x U16 -> U32 is only possible for 1 prec combination
        if [[ ("$TYPE" == '6') && ("$PREC" != 'U16_U16_U32_IMPLICIT') ]]; then
          continue
        fi
        if [[ ("$TYPE" != '6') && ("$PREC" == 'U16_U16_U32_IMPLICIT') ]]; then
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
        elif [ "$TYPE" == '6' ] ; then
          TPPNAME="zip"
        elif [ "$TYPE" == '9' ] ; then
          TPPNAME="max"
        elif [ "$TYPE" == '10' ] ; then
          TPPNAME="min"
        elif [ "$TYPE" == '27' ] ; then
          TPPNAME="cmp_gt"
        elif [ "$TYPE" == '28' ] ; then
          TPPNAME="cmp_ge"
        elif [ "$TYPE" == '29' ] ; then
          TPPNAME="cmp_lt"
        elif [ "$TYPE" == '30' ] ; then
          TPPNAME="cmp_le"
        elif [ "$TYPE" == '31' ] ; then
          TPPNAME="cmp_eq"
        elif [ "$TYPE" == '32' ] ; then
          TPPNAME="cmp_ne"
        else
          continue
        fi

        if [[ ("$TYPE" == '27') || ("$TYPE" == '28') || ("$TYPE" == '29') || ("$TYPE" == '30') || ("$TYPE" == '31') || ("$TYPE" == '32') ]]; then
          if [ "$ROUND" == 'stoch' ]; then
            continue
          fi
          PREC_IN0=$(echo "$PRECLC" |  awk -F"_" '{print $1}')
          PREC_IN1=$(echo "$PRECLC" |  awk -F"_" '{print $2}')
          if [[ ("$PREC_IN0" == 'f64') || ("$PREC_IN1" == 'f64') ]]; then
            continue
          fi
          PRECH_IN0=$(echo "$PREC" |  awk -F"_" '{print $1}')
          PRECH_IN1=$(echo "$PREC" |  awk -F"_" '{print $2}')
          PRECLC=${PREC_IN0}_${PREC_IN1}_implicit_f32
          PRECSCRIPT=${PRECH_IN0}_${PRECH_IN1}_IMPLICIT_F32
        else
          if [ "$ROUND" == 'stoch' ]; then
            PREC_OUT=$(echo "$PRECLC" |  awk -F"_" '{print $3}')
            if [ "$PREC_OUT" == 'bf8' ] ; then
              PREC_IN0=$(echo "$PRECLC" |  awk -F"_" '{print $1}')
              PREC_IN1=$(echo "$PRECLC" |  awk -F"_" '{print $2}')
              PREC_COMP=$(echo "$PRECLC" |  awk -F"_" '{print $4}')
              PREC_OUT=${PREC_OUT}${ROUND}
              RMODE=1
              PRECLC=${PREC_IN0}_${PREC_IN1}_${PREC_OUT}_${PREC_COMP}
            else
              continue
            fi
          fi
        fi

        OUTNAME=${OUTNAME}${TPPNAME}_${PRECLC}_${LD}.sh

        # generate script by sed
        sed "s/PREC=0/PREC=\"${PRECSCRIPT}\"/g" ${HERE}/binary.tpl \
        | sed "s/BINARY_OP=0/BINARY_OP=${TYPE}/g" \
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
