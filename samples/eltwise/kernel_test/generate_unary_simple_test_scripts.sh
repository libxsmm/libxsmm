#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'F32_F32_F32' 'BF16_BF16_BF16' 'BF16_BF16_F32' 'F32_BF16_F32' 'BF16_F32_F32' 'F16_F16_F16' 'F16_F16_F32' 'F32_F16_F32' 'F16_F32_F32' 'BF8_BF8_BF8' 'BF8_BF8_F32' 'F32_BF8_F32' 'BF8_F32_F32' 'HF8_HF8_HF8' 'HF8_HF8_F32' 'F32_HF8_F32' 'HF8_F32_F32' 'F64_F64_F64'; do
  for TYPE in 1 2 3 4 7 8 9 10 11 12 13 14 15 16 17 27 64 65; do
    for LD in 'eqld' 'gtld'; do
      TPPNAME="none"
      OUTNAME="unary_"
      PRECLC=$(echo "$PREC" | awk '{print tolower($0)}')

      # approximations are not supportted for the time being in FP64
      if [[ (("$TYPE" == '7') || ("$TYPE" == '8') || ("$TYPE" == '9') || ("$TYPE" == '10') || ("$TYPE" == '11') || ("$TYPE" == '12') || ("$TYPE" == '17') || ("$TYPE" == '27')) && ("$PREC" == 'F64_F64_F64') ]]; then
        continue
      fi

      # only cpy TPP has low precision compute
      if [[ ("$TYPE" != '1') && ("$TYPE" != '2') && ("$TYPE" != '27') && (("$PREC" == 'F16_F16_F16') || ("$PREC" == 'BF16_BF16_BF16') || ("$PREC" == 'BF8_BF8_BF8') || ("$PREC" == 'HF8_HF8_HF8')) ]]; then
        continue
      fi

      # decomp FP32 -> BF16 in only possible for 1 prec combination
      if [[ (("$TYPE" == '64') || ("$TYPE" == '65')) && ("$PREC" != 'F32_BF16_F32') ]]; then
        continue
      fi

      # get TPP name
      if [ "$TYPE" == '1' ] ; then
        TPPNAME="copy"
      elif [ "$TYPE" == '2' ] ; then
        TPPNAME="xor"
      elif [ "$TYPE" == '3' ] ; then
        TPPNAME="x2"
      elif [ "$TYPE" == '4' ] ; then
        TPPNAME="sqrt"
      elif [ "$TYPE" == '7' ] ; then
        TPPNAME="tanh"
      elif [ "$TYPE" == '8' ] ; then
        TPPNAME="tanh_inv"
      elif [ "$TYPE" == '9' ] ; then
        TPPNAME="sigmoid"
      elif [ "$TYPE" == '10' ] ; then
        TPPNAME="sigmoid_inv"
      elif [ "$TYPE" == '11' ] ; then
        TPPNAME="gelu"
      elif [ "$TYPE" == '12' ] ; then
        TPPNAME="gelu_inv"
      elif [ "$TYPE" == '13' ] ; then
        TPPNAME="negate"
      elif [ "$TYPE" == '14' ] ; then
        TPPNAME="inc"
      elif [ "$TYPE" == '15' ] ; then
        TPPNAME="rcp"
      elif [ "$TYPE" == '16' ] ; then
        TPPNAME="rsqrt"
      elif [ "$TYPE" == '17' ] ; then
        TPPNAME="exp"
      elif [ "$TYPE" == '27' ] ; then
        TPPNAME="replicate_col_var"
      elif [ "$TYPE" == '64' ] ; then
        TPPNAME="decomp_fp32_to_bf16x2"
      elif [ "$TYPE" == '65' ] ; then
        TPPNAME="decomp_fp32_to_bf16x3"
      else
        continue
      fi

      OUTNAME=${OUTNAME}${TPPNAME}_${PRECLC}_${LD}.sh

      # generate script by sed
      sed "s/PREC=0/PREC=\"${PREC}\"/g" unary_simple.tpl \
      | sed "s/UNARY_OP=0/UNARY_OP=${TYPE}/g" \
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
