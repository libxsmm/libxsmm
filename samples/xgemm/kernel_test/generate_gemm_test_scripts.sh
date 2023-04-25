#!/usr/bin/env bash

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=18
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for BINARY_POSTOP in 0 1; do
  for UNARY_POSTOP in 0 1 2 3; do
    for PREC in 'F64' 'F32' 'BF16' 'I16I32' 'USI8I32' 'SUI8I32' 'USI8F32' 'SUI8F32' 'BF16F32' 'BF16_FLAT' 'BF16F32_FLAT' 'BF8' 'BF8F32' 'HF8' 'HF8F32' 'BF8_FLAT' 'BF8F32_FLAT' 'HF8_FLAT' 'HF8F32_FLAT'; do
      for LD in 'eqld' 'gtld'; do
        for CVNNI in 0 1; do
          for TRA in 0 1; do
            for TRB in 0 1; do
              for STACK in 0 1; do
                # check if the current experiment has eltwise fusion
                NOFUSION=0
                MSTART=1
                NSTART=1
                KSTART=1
                MSTEP=1
                NSTEP=1
                KSTEP=1

                if [[ ("$BINARY_POSTOP" == '0'  &&  "$UNARY_POSTOP" == '0') && ( "$CVNNI" == '0') ]]; then
                  NOFUSION=1
                fi

                # TODO: all the "continue" ifs should be handled by allow list outside of this script
                # transpose A AND B at the same time is right now not supported
                if [[ ( "$TRA" == '1'  &&  "$TRB" == '1') ]]; then
                  continue
                fi

                # some precision don't support fusion at all
                if [[ ( "$PREC" == 'F64' || "$PREC" == 'I16I32' || "$PREC" == 'USI8I32' || "$PREC" == 'SUI8I32' || "$PREC" == 'USI8F32' || "$PREC" == 'SUI8F32' ) && ( "$NOFUSION" == '0' ) ]] ; then
                  continue
                fi

                # we only test storing in VNNI layout when there is no other fusion
                if [[ ("$BINARY_POSTOP" != '0'  ||  "$UNARY_POSTOP" != '0') && ( "$CVNNI" == '1') ]]; then
                  continue
                fi

                # storing in VNNI layout is only supported/makes sense for select precision
                if [[ ("$PREC" == 'F32' || "$PREC" == 'BF16F32' || "$PREC" == 'BF16_FLAT' || "$PREC" == 'BF16F32_FLAT' || "$PREC" == 'BF8F32' || "$PREC" == 'BF8_FLAT' || "$PREC" == 'BF8F32_FLAT' || "$PREC" == 'HF8F32' || "$PREC" == 'HF8_FLAT' || "$PREC" == 'HF8F32_FLAT' ) && ( "$CVNNI" == '1' ) ]]; then
                  continue
                fi

                # low precision has no transpose support
                if [[ ( "$PREC" == 'I16I32' || "$PREC" == 'USI8I32' || "$PREC" == 'SUI8I32' || "$PREC" == 'USI8F32' || "$PREC" == 'SUI8F32' || "$PREC" == 'BF16' || "$PREC" == 'BF16_FLAT' || "$PREC" == 'BF16F32' || "$PREC" == 'BF16F32_FLAT' || "$PREC" == 'BF8' || "$PREC" == 'BF8_FLAT' || "$PREC" == 'BF8F32' || "$PREC" == 'BF8F32_FLAT'|| "$PREC" == 'HF8' || "$PREC" == 'HF8_FLAT' || "$PREC" == 'HF8F32' || "$PREC" == 'HF8F32_FLAT'  ) && ( "$TRA" == '1'  ||  "$TRB" == '1') ]]; then
                  continue
                fi

                # only FP8 has stack tansforms for data prep
                if [[ ( "$PREC" != 'BF8' && "$PREC" != 'BF8_FLAT' && "$PREC" != 'BF8F32' && "$PREC" != 'BF8F32_FLAT' && "$PREC" != 'HF8' && "$PREC" != 'HF8_FLAT' && "$PREC" != 'HF8F32' && "$PREC" != 'HF8F32_FLAT') && ( "$STACK" == '1' ) ]]; then
                  continue
                fi

                if [ "$PREC" == 'F64' ] ; then
                  OUTNAME="dgemm_"
                elif [ "$PREC" == 'F32' ] ; then
                  OUTNAME="sgemm_"
                elif [ "$PREC" == 'I16I32' ] ; then
                  OUTNAME="i16i32gemm_"
                  KSTART=2
                  KSTEP=2
                elif [ "$PREC" == 'USI8I32' ] ; then
                  OUTNAME="usi8i32gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'SUI8I32' ] ; then
                  OUTNAME="sui8i32gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'USI8F32' ] ; then
                  OUTNAME="usi8f32gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'SUI8F32' ] ; then
                  OUTNAME="sui8f32gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'BF16' ] ; then
                  OUTNAME="bf16gemm_"
                  KSTART=2
                  KSTEP=2
                elif [ "$PREC" == 'BF16F32' ] ; then
                  OUTNAME="bf16f32gemm_"
                  KSTART=2
                  KSTEP=2
                elif [ "$PREC" == 'BF16_FLAT' ] ; then
                  OUTNAME="bf16_flatgemm_"
                  KSTART=2
                  KSTEP=2
                elif [ "$PREC" == 'BF16F32_FLAT' ] ; then
                  OUTNAME="bf16f32_flatgemm_"
                  KSTART=2
                  KSTEP=2
                elif [ "$PREC" == 'BF8' ] ; then
                  OUTNAME="bf8gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'BF8F32' ] ; then
                  OUTNAME="bf8f32gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'HF8' ] ; then
                  OUTNAME="hf8gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'HF8F32' ] ; then
                  OUTNAME="hf8f32gemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'BF8_FLAT' ] ; then
                  OUTNAME="bf8_flatgemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'BF8F32_FLAT' ] ; then
                  OUTNAME="bf8f32_flatgemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'HF8_FLAT' ] ; then
                  OUTNAME="hf8_flatgemm_"
                  KSTART=4
                  KSTEP=4
                elif [ "$PREC" == 'HF8F32_FLAT' ] ; then
                  OUTNAME="hf8f32_flatgemm_"
                  KSTART=4
                  KSTEP=4
                 else
                  continue
                fi

                if [ "$CVNNI" == '1' ] ; then
                  if [ "$PREC" == 'BF16' ] ; then
                    NSTART=2
                    NSTEP=2
                  elif [ "$PREC" == 'BF8' ] ; then
                    NSTART=4
                    NSTEP=4
                  elif [ "$PREC" == 'HF8' ] ; then
                    NSTART=4
                    NSTEP=4
                  else
                    continue
                  fi
                fi

                if [ "$STACK" == '1' ] ; then
                  OUTNAME=$OUTNAME"via_stack_"
                fi

                if [ "$TRA" == '0' ] ; then
                  OUTNAME=$OUTNAME"n"
                else
                  OUTNAME=$OUTNAME"t"
                fi

                if [ "$TRB" == '0' ] ; then
                  OUTNAME=$OUTNAME"n_"$LD
                else
                  OUTNAME=$OUTNAME"t_"$LD
                fi

                if [ "$NOFUSION" == '0' ] ; then
                  OUTNAME=$OUTNAME"_bop"$BINARY_POSTOP"_uop"$UNARY_POSTOP"_cvnni"$CVNNI".slurm"
                else
                  OUTNAME=$OUTNAME".slurm"
                fi

                #echo "Copying "$TPLNAME" to "$OUTNAME
                sed "s/PREC=0/PREC=\"${PREC}\"/g" gemm_kernel_fused.tpl \
                | sed "s/TRA=0/TRA=${TRA}/g" \
                | sed "s/TRB=0/TRB=${TRB}/g" \
                | sed "s/BINARY_POSTOP=0/BINARY_POSTOP=${BINARY_POSTOP}/g" \
                | sed "s/UNARY_POSTOP=0/UNARY_POSTOP=${UNARY_POSTOP}/g" \
                | sed "s/CVNNI=0/CVNNI=${CVNNI}/g" \
                | sed "s/MSTART/${MSTART}/g" \
                | sed "s/MSTEP/${MSTEP}/g" \
                | sed "s/NSTART/${NSTART}/g" \
                | sed "s/NSTEP/${NSTEP}/g" \
                | sed "s/KSTART/${KSTART}/g" \
                | sed "s/KSTEP/${KSTEP}/g" \
                | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
                >${OUTNAME}

                # for gt we need to touch up the script
                if [ "$LD" == 'gtld' ] ; then
                  sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ '100 100 100'/g" ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                fi

                # trB we need to switch LDB
                if [ "$TRA" == '1' ] ; then
                  sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(k) + ' ' + str(k) + ' ' + str(m)/g" ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                fi

                # trB we need to switch LDB
                if [ "$TRB" == '1' ] ; then
                  sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' ' + str(n) + ' ' + str(m)/g" ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                fi

                # remove env variable
                if [ "$STACK" == '0' ] ; then
                  sed "/export LIBXSMM_ENV_VAR=1/d" ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                fi

                # stack exports
                if [[ ( "$PREC" == 'BF8' || "$PREC" == 'BF8_FLAT' || "$PREC" == 'BF8F32' || "$PREC" == 'BF8F32_FLAT' ) && ( "$STACK" == '1' ) ]]; then
                  sed 's/LIBXSMM_ENV_VAR/LIBXSMM_BF8_GEMM_VIA_STACK/g' ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                fi
                if [[ ( "$PREC" == 'HF8' || "$PREC" == 'HF8_FLAT' || "$PREC" == 'HF8F32' || "$PREC" == 'HF8F32_FLAT' ) && ( "$STACK" == '1' ) ]]; then
                  sed 's/LIBXSMM_ENV_VAR/LIBXSMM_HF8_GEMM_VIA_STACK/g' ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                fi

                # nofusion, we use the regular kernel
                if [ "$NOFUSION" == '1' ] ; then
                  sed 's/gemm_kernel_fused/gemm_kernel/g' ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                  sed 's/ ${BINARY_POSTOP} ${UNARY_POSTOP} ${CVNNI}//g' ${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} ${OUTNAME}
                fi

                chmod 755 ${OUTNAME}

                # for certain prericsion we are adding some special scripts for more restricted hardware
                # QVNNI for I16I32
                if [ "$PREC" == 'I16I32' ] ; then
                  cp ${OUTNAME} qvnni_${OUTNAME}
                  sed 's/randnumk = rnd.sample(range(2,101,2), .*)/randnumk = rnd.sample(range(8,101,8), 8)/g' qvnni_${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} qvnni_${OUTNAME}
                  chmod 755 qvnni_${OUTNAME}
                fi

                # MMLA for BF16
                if [[ ( "$PREC" == 'BF16' || "$PREC" == 'BF16F32' ) ]] ; then
                  cp ${OUTNAME} mmla_${OUTNAME}
                  sed 's/randnumk = rnd.sample(range(2,101,2)/randnumk = rnd.sample(range(4,101,4)/g' mmla_${OUTNAME} >${TMPFILE}
                  cp ${TMPFILE} mmla_${OUTNAME}
                  if [ "$CVNNI" == '1' ] ; then
                    sed 's/randnumn = rnd.sample(range(2,101,2)/randnumn = rnd.sample(range(4,101,4)/g' mmla_${OUTNAME} >${TMPFILE}
                    cp ${TMPFILE} mmla_${OUTNAME}
                  fi
                  chmod 755 mmla_${OUTNAME}

                  # create MMLA scripts with B in VNNIT
                  NEWNAME=mmla_bvnni_${OUTNAME}
                  NEWNAME="${NEWNAME/_nn_/_nt_}"
                  sed \
                    -e "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' ' + str(n) + ' ' + str(m)/g" \
                    -e "s/PREC=\"${PREC}\"/PREC=\"${PREC}_BVNNI\"/g" \
                    -e 's/TRB=0/TRB=1/g' \
                    mmla_${OUTNAME} >${NEWNAME}
                  chmod 755 ${NEWNAME}
                fi

                # BFDOT for BF16 wth B in VNNIT
                if [[ ( "$PREC" == 'BF16' || "$PREC" == 'BF16F32' ) ]] ; then
                  # create BFDOT scripts with B in VNNIT
                  NEWNAME=bfdot_bvnni_${OUTNAME}
                  NEWNAME="${NEWNAME/_nn_/_nt_}"
                  sed \
                    -e "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' ' + str(n) + ' ' + str(m)/g" \
                    -e "s/PREC=\"${PREC}\"/PREC=\"${PREC}_BVNNI\"/g" \
                    -e 's/TRB=0/TRB=1/g' \
                    ${OUTNAME} >${NEWNAME}
                  chmod 755 ${NEWNAME}
                fi
              done
            done
          done
        done
      done
    done
  done
done
