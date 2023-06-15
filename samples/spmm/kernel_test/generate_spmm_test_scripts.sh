#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=4
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for PREC in 'F32_F32_F32_F32' 'BF16_BF16_F32_BF16' 'U8_I8_I32_I32' 'I8_U8_I32_I32'; do
  for AVNNI in 0 1; do
    for BVNNI in 0 1; do
      for TRB in 0 1; do
        NSTART=1
        KSTART=1
        MSTART=1
        MSTEP=1
        BN_LIST=(1)
        BK_LIST=(1)

        if [[ ("$PREC" == 'F32_F32_F32_F32') && ( "$AVNNI" == '1' || "$BVNNI" == '1' || "$TRB" == '1') ]]; then
          continue
        fi
        if [[ ("$PREC" != 'F32_F32_F32_F32') && ( "$AVNNI" == '0' ) ]]; then
          continue
        fi
        if [[ ("$PREC" != 'F32_F32_F32_F32') && ( "$BVNNI" == '1' && "$TRB" == '0') ]]; then
          continue
        fi
        if [[ ("$PREC" != 'F32_F32_F32_F32') && ( "$BVNNI" == '0' && "$TRB" == '1') ]]; then
          continue
        fi

        if [ "$PREC" == 'F32_F32_F32_F32' ] ; then
          OUTNAME="spmm_f32_avnni0_bvnni0_trb0"
          BN_LIST=(1 2 4 8 16 32)
          BK_LIST=(1 2 4 8 16 32)
        elif [ "$PREC" == 'BF16_BF16_F32_BF16' ] ; then
          OUTNAME="spmm_bf16"
          if [[ ("$BVNNI" == '1') && ("$TRB" == '1') ]]; then
            OUTNAME=$OUTNAME"_avnni1_bvnni1_trb1"
            BN_LIST=(1 2 4 8 16 32)
            BK_LIST=(4 8 16 32)
          else
            OUTNAME=$OUTNAME"_avnni1_bvnni0_trb0"
            BN_LIST=(1 2 4 8 16 32)
            BK_LIST=(2 4 8 16 32)
          fi
        elif [[ ("$PREC" == 'U8_I8_I32_I32') || ("$PREC" == 'I8_U8_I32_I32') ]] ; then
          if [ "$PREC" == 'U8_I8_I32_I32' ] ; then
            OUTNAME="spmm_u8i8i32"
          fi
          if [ "$PREC" == 'I8_U8_I32_I32' ] ; then
            OUTNAME="spmm_i8u8i32"
          fi
          if [[ ("$BVNNI" == '1') && ("$TRB" == '1') ]]; then
            OUTNAME=$OUTNAME"_avnni1_bvnni1_trb1"
            BN_LIST=(1 2 4 8 16 32)
            BK_LIST=(8 16 32)
          else
            OUTNAME=$OUTNAME"_avnni1_bvnni0_trb0"
            BN_LIST=(1 2 4 8 16 32)
            BK_LIST=(4 8 16 32)
          fi
        fi

        for KSTEP in "${BK_LIST[@]}"; do
          for NSTEP in "${BN_LIST[@]}"; do
            OUTNAMEFINAL=$OUTNAME"_bk"$KSTEP"_bn"$NSTEP".slurm"
            sed "s/PREC=0/PREC=\"${PREC}\"/g" spmm_kernel.tpl \
            | sed "s/TRB=0/TRB=${TRB}/g" \
            | sed "s/AVNNI=0/AVNNI=${AVNNI}/g" \
            | sed "s/BVNNI=0/BVNNI=${BVNNI}/g" \
            | sed "s/MSTART/${MSTART}/g" \
            | sed "s/MSTEP/${MSTEP}/g" \
            | sed "s/NSTART/${NSTEP}/g" \
            | sed "s/NSTEP/${NSTEP}/g" \
            | sed "s/KSTART/${KSTEP}/g" \
            | sed "s/KSTEP/${KSTEP}/g" \
            | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
              >${OUTNAMEFINAL}
            #echo "Outname us ${OUTNAME} : kstep is ${KSTEP} and nstep is ${NSTEP} and prec is ${PREC}"
          done
        done
      done
    done
  done
done

