#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=10
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'F32_F32_F32_F32' 'BF16_BF16_F32_F32' 'BF16_BF16_F32_BF16' 'BF8_BF16_F32_F32' 'BF8_BF16_F32_BF16' 'HF8_BF16_F32_F32' 'HF8_BF16_F32_BF16'; do
  for LD in 'eqld' 'gtld'; do
    for AVNNI in 0 1; do
      # check if the current experiment has eltwise fusion
      NSTART=1
      KSTART=32
      NSTEP=1
      KSTEP=32

      if [[ -z "${SSIZE}" ]]; then
        SAMPLESIZE=10
      else
        SAMPLESIZE=${SSIZE}
      fi

      # loading A in VNNI layout is only supported/makes sense for select precision
      if [[ ("$PREC" == 'F32_F32_F32_F32') && ( "$AVNNI" == '1' ) ]]; then
        continue
      fi

      # loading A in VNNI layout is mandatory for select precision
      if [[ ("$PREC" == 'BF16_BF16_F32_F32' || "$PREC" == 'BF16_BF16_F32_BF16' || "$PREC" == 'BF8_BF16_F32_F32' || "$PREC" == 'BF8_BF16_F32_BF16' || "$PREC" == 'HF8_BF16_F32_F32' || "$PREC" == 'HF8_BF16_F32_BF16') && ( "$AVNNI" == '0' ) ]]; then
        continue
      fi

      if [ "$PREC" == 'F32_F32_F32_F32' ] ; then
        OUTNAME="fp32_spmm_"
      elif [ "$PREC" == 'BF16_BF16_F32_F32' ] ; then
        OUTNAME="bf16f32_spmm_"
      elif [ "$PREC" == 'BF16_BF16_F32_BF16' ] ; then
        OUTNAME="bf16bf16_spmm_"
      elif [ "$PREC" == 'BF8_BF16_F32_F32' ] ; then
        OUTNAME="bf8bf16f32_spmm_"
      elif [ "$PREC" == 'BF8_BF16_F32_BF16' ] ; then
        OUTNAME="bf8bf16bf16_spmm_"
      elif [ "$PREC" == 'HF8_BF16_F32_F32' ] ; then
        OUTNAME="hf8bf16f32_spmm_"
      elif [ "$PREC" == 'HF8_BF16_F32_BF16' ] ; then
        OUTNAME="hf8bf16bf16_spmm_"
      else
        continue
      fi

      OUTNAME=$OUTNAME$LD.slurm

      #echo "Copying "$TPLNAME" to "$OUTNAME
      sed "s/PREC=0/PREC=\"${PREC}\"/g" ${HERE}/spmm_kernel.tpl \
      | sed "s/AVNNI=0/AVNNI=${AVNNI}/g" \
      | sed "s/NSTEP/${NSTEP}/g" \
      | sed "s/KSTART/${KSTART}/g" \
      | sed "s/KSTEP/${KSTEP}/g" \
      | sed "s/NSTART/${NSTART}/g" \
      | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
      >${HERE}/${OUTNAME}

      # for gt we need to touch up the script
      if [ "$LD" == 'gtld' ] ; then
        sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' 2048 100'/g" ${HERE}/${OUTNAME} >${TMPFILE}
        cp ${TMPFILE} ${HERE}/${OUTNAME}
      fi

      chmod 755 ${HERE}/${OUTNAME}

    done
  done
done

