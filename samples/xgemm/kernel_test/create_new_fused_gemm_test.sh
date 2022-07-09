#!/usr/bin/env bash

for BINARY_POSTOP in 0 1; do
  for UNARY_POSTOP in 0 1 2 3; do
    for PREC in 'SP' 'BF16' 'BF16F32' 'BF16_FLAT' 'BF16F32_FLAT' 'BF8' 'BF8F32'; do
      for LD in 'eqld' 'gtld'; do
        for CVNNI in 0 1; do
          if [[ ("$BINARY_POSTOP" == '0'  &&  "$UNARY_POSTOP" == '0') && ( "$CVNNI" == '0') ]]; then
            continue
          fi
          if [[ ("$BINARY_POSTOP" != '0'  ||  "$UNARY_POSTOP" != '0') && ( "$CVNNI" == '1') ]]; then
            continue
          fi
          if [[ ("$PREC" == 'SP' &&  "$CVNNI" == '1' ) ]]; then
            continue
          fi
          if [[ ("$PREC" == 'BF16F32'  &&  "$CVNNI" == '1') ]]; then
            continue
          fi
          if [[ ("$PREC" == 'BF16_FLAT'  && "$CVNNI" == '1') ]]; then
            continue
          fi
          if [[ ("$PREC" == 'BF16F32_FLAT' && "$CVNNI" == '1') ]]; then
            continue
          fi

          for TRA in 0 1; do
            for TRB in 0 1; do
              if [[ (( "$PREC" == 'BF16_FLAT'  ||  "$PREC" == 'BF16' ) || ( "$PREC" == 'BF16F32'  ||  "$PREC" == 'BF16F32_FLAT') || "$PREC" == 'BF8'  ||  "$PREC" == 'BF8F32' ) && ( "$TRA" == '1'  ||  "$TRB" == '1') ]]; then
                continue
              fi

              if [[ ( "$TRA" == '1'  &&  "$TRB" == '1') ]]; then
                continue
              fi

              if [ "$PREC" == 'SP' ] ; then
                TPLNAME="sgemm_"
              elif [ "$PREC" == 'BF16' ] ; then
                TPLNAME="bf16gemm_"
              elif [ "$PREC" == 'BF16F32' ] ; then
                TPLNAME="bf16f32gemm_"
              elif [ "$PREC" == 'BF16_FLAT' ] ; then
                TPLNAME="bf16_flatgemm_"
              elif [ "$PREC" == 'BF16F32_FLAT' ] ; then
                TPLNAME="bf16f32_flatgemm_"
              elif [ "$PREC" == 'BF8' ] ; then
                TPLNAME="bf8gemm_"
              elif [ "$PREC" == 'BF8F32' ] ; then
                TPLNAME="bf8f32gemm_"
              else
                continue
              fi

              if [ "$TRA" == '0' ] ; then
                TPLNAME=$TPLNAME"n"
              else
                TPLNAME=$TPLNAME"t"
              fi

              if [ "$TRB" == '0' ] ; then
                TPLNAME=$TPLNAME"n_"$LD
              else
                TPLNAME=$TPLNAME"t_"$LD
              fi

              OUTNAME=$TPLNAME"_bop"$BINARY_POSTOP"_uop"$UNARY_POSTOP"_cvnni"$CVNNI".slurm"
              TPLNAME=$TPLNAME"_tpl"

              #echo "Copying "$TPLNAME" to "$OUTNAME
              cp ${TPLNAME} ${OUTNAME}
              sed "s/PREC=\"SP\"/PREC=\"${PREC}\"/g" -i ${OUTNAME}
              sed "s/TRA=0/TRA=${TRA}/g" -i ${OUTNAME}
              sed "s/TRB=0/TRB=${TRB}/g" -i ${OUTNAME}
              sed "s/BINARY_POSTOP=0/BINARY_POSTOP=${BINARY_POSTOP}/g" -i ${OUTNAME}
              sed "s/UNARY_POSTOP=0/UNARY_POSTOP=${UNARY_POSTOP}/g" -i ${OUTNAME}
              sed "s/CVNNI=0/CVNNI=${CVNNI}/g" -i ${OUTNAME}

            done
          done
        done
      done
    done
  done
done
