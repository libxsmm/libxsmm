#!/usr/bin/env bash

HERE=$(cd "$(dirname "$0")" && pwd -P)

if [[ -z "${SSIZE}" ]]; then
  SAMPLESIZE=10
else
  SAMPLESIZE=${SSIZE}
fi

TMPFILE=$(mktemp)
trap 'rm ${TMPFILE}' EXIT

for BINARY_POSTOP in 0 1; do
  for UNARY_POSTOP in 0 1 2 3; do
    for PREC in 'F64_F64_F64_F64' 'F32_F32_F32_F32' 'BF16_BF16_F32_F32' 'BF16_BF16_F32_BF16' 'BF8_BF8_F32_F32' 'BF8_BF8_F32_BF8' 'HF8_HF8_F32_F32' 'HF8_HF8_F32_HF8' 'I16_I16_I32_I32' 'U8_I8_I32_I32' 'I8_U8_I32_I32' 'U8_I8_I32_F32' 'I8_U8_I32_F32' 'F16_F16_F16_F16' 'I8_F16_F16_F16' 'BF8_F16_F16_F16' 'F16_F16_F32_F16' 'I8_F16_F32_F16' 'BF8_F16_F32_F16' 'F16_F16_IMPLICIT_F16' 'I8_F16_IMPLICIT_F16' 'BF8_F16_IMPLICIT_F16' 'F16_F16_F16_F32' 'I8_F16_F16_F32' 'BF8_F16_F16_F32' 'F16_F16_F32_F32' 'I8_F16_F32_F32' 'BF8_F16_F32_F32' 'F16_F16_IMPLICIT_F32' 'I8_F16_IMPLICIT_F32' 'BF8_F16_IMPLICIT_F32' 'I8_BF16_F32_F32' 'I8_BF16_F32_BF16' 'I4_F16_IMPLICIT_F16' 'I4_F16_F32_F16' 'I4_F16_F16_F16' 'I4_F16_IMPLICIT_F32' 'I4_F16_F16_F32' 'I4_F16_F32_F32' 'U4_F16_IMPLICIT_F16' 'U4_F16_F32_F16' 'U4_F16_F16_F16' 'U4_F16_IMPLICIT_F32' 'U4_F16_F16_F32' 'U4_F16_F32_F32' 'U8_F16_F16_F16' 'U8_F16_F32_F16' 'U8_F16_IMPLICIT_F16' 'U8_F16_F16_F32' 'U8_F16_F32_F32' 'U8_F16_IMPLICIT_F32' 'U8_BF16_F32_F32' 'U8_BF16_F32_BF16' ; do
      for LD in 'eqld' 'gtld'; do
        for AVNNI in 0 1; do
          for BVNNI in 0 1; do
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

                    #split PREC string into 4 variables


                    if [[ ("$BINARY_POSTOP" == '0'  &&  "$UNARY_POSTOP" == '0') ]]; then
                      NOFUSION=1
                    fi

                    # TODO: all the "continue" ifs should be handled by allow list outside of this script
                    # transpose A AND B at the same time is right now not supported
                    if [[ ( "$TRA" == '1'  &&  "$TRB" == '1') ]]; then
                      continue
                    fi

                    # TODO fix BVNNI supprt in the script
                    if [ "$BVNNI" == '1' ] ; then
                      continue
                    fi

                    # some precision don't support fusion at all
                    if [[ ( "$PREC" == 'F64_F64_F64_F64' || "$PREC" == 'I16_I16_I32_I32' || "$PREC" == 'U8_I8_I32_I32' || "$PREC" == 'I8_U8_I32_I32' || "$PREC" == 'U8_I8_I32_F32' || "$PREC" == 'I8_U8_I32_F32' || "$PREC" == 'F16_F16_IMPLICIT_F16' || "$PREC" == 'I8_F16_IMPLICIT_F16' || "$PREC" == 'F16_F16_F16_F16' || "$PREC" == 'I8_F16_F16_F16' || "$PREC" == 'I8_F16_F32_F16' || "$PREC" == 'F16_F16_IMPLICIT_F32' || "$PREC" == 'I8_F16_IMPLICIT_F32' || "$PREC" == 'F16_F16_F16_F32' || "$PREC" == 'I8_F16_F16_F32'  || "$PREC" == 'I8_F16_F32_F32' || "$PREC" == 'I8_BF16_F32_F32' || "$PREC" == 'I8_BF16_F32_BF16' || "$PREC" == 'BF8_F16_IMPLICIT_F16' || "$PREC" == 'BF8_F16_F32_F16' || "$PREC" == 'BF8_F16_F16_F16' || "$PREC" == 'BF8_F16_IMPLICIT_F32' || "$PREC" == 'BF8_F16_F16_F32' || "$PREC" == 'BF8_F16_F32_F32' || "$PREC" == 'I4_F16_IMPLICIT_F16' || "$PREC" == 'I4_F16_F32_F16' || "$PREC" == 'I4_F16_F16_F16' || "$PREC" == 'I4_F16_IMPLICIT_F32' || "$PREC" == 'I4_F16_F16_F32' || "$PREC" == 'I4_F16_F32_F32' || "$PREC" == 'U4_F16_IMPLICIT_F16' || "$PREC" == 'U4_F16_F32_F16' || "$PREC" == 'U4_F16_F16_F16' || "$PREC" == 'U4_F16_IMPLICIT_F32' || "$PREC" == 'U4_F16_F16_F32' || "$PREC" == 'U4_F16_F32_F32' || "$PREC" == 'U8_F16_F16_F16' || "$PREC" == 'U8_F16_F32_F16' || "$PREC" == 'U8_F16_IMPLICIT_F32' || "$PREC" == 'U8_F16_F16_F32' || "$PREC" == 'U8_F16_F32_F32' || "$PREC" == 'U8_BF16_F32_F32' || "$PREC" == 'U8_BF16_F32_BF16' || "$PREC" == 'U8_F16_IMPLICIT_F16' ) && ( "$NOFUSION" == '0' ) ]] ; then
                      continue
                    fi

                    # we only test storing in VNNI layout when there is no other fusion
                    if [[ ("$BINARY_POSTOP" != '0'  ||  "$UNARY_POSTOP" != '0') && ( "$CVNNI" == '1') ]]; then
                      continue
                    fi

                    # loading A in VNNI layout is only supported/makes sense for select precision
                    if [[ ("$PREC" == 'F64_F64_F64_F64' || "$PREC" == 'F32_F32_F32_F32' || "$PREC" == 'F16_F16_IMPLICIT_F16' || "$PREC" == 'I8_F16_IMPLICIT_F16' || "$PREC" == 'F16_F16_F16_F16' || "$PREC" == 'I8_F16_F16_F16' || "$PREC" == 'I8_F16_F32_F16' || "$PREC" == 'F16_F16_IMPLICIT_F32' || "$PREC" == 'I8_F16_IMPLICIT_F32' || "$PREC" == 'F16_F16_F16_F32' || "$PREC" == 'I8_F16_F16_F32'  || "$PREC" == 'I8_F16_F32_F32' || "$PREC" == 'I8_BF16_F32_F32' || "$PREC" == 'I8_BF16_F32_BF16' || "$PREC" == 'BF8_F16_IMPLICIT_F16' || "$PREC" == 'BF8_F16_F32_F16' || "$PREC" == 'BF8_F16_F16_F16' || "$PREC" == 'BF8_F16_IMPLICIT_F32' || "$PREC" == 'BF8_F16_F16_F32' || "$PREC" == 'BF8_F16_F32_F32' || "$PREC" == 'I4_F16_IMPLICIT_F16' || "$PREC" == 'I4_F16_F32_F16' || "$PREC" == 'I4_F16_F16_F16' || "$PREC" == 'I4_F16_IMPLICIT_F32' || "$PREC" == 'I4_F16_F16_F32' || "$PREC" == 'I4_F16_F32_F32' || "$PREC" == 'U4_F16_IMPLICIT_F16' || "$PREC" == 'U4_F16_F32_F16' || "$PREC" == 'U4_F16_F16_F16' || "$PREC" == 'U4_F16_IMPLICIT_F32' || "$PREC" == 'U4_F16_F16_F32' || "$PREC" == 'U4_F16_F32_F32' || "$PREC" == 'U8_F16_F16_F16' || "$PREC" == 'U8_F16_F32_F16' || "$PREC" == 'U8_F16_IMPLICIT_F32' || "$PREC" == 'U8_F16_F16_F32' || "$PREC" == 'U8_F16_F32_F32' || "$PREC" == 'U8_BF16_F32_F32' || "$PREC" == 'U8_BF16_F32_BF16' || "$PREC" == 'U8_F16_IMPLICIT_F16' ) && ( "$AVNNI" == '1' ) ]]; then
                      continue
                    fi

                    # loading B in VNNI layout is only supported/makes sense for select precision
                    if [[ ("$PREC" == 'F64_F64_F64_F64' || "$PREC" == 'F32_F32_F32_F32' || "$PREC" == 'I16_I16_I32_I32' || "$PREC" == 'U8_I8_I32_I32' || "$PREC" == 'I8_U8_I32_I32' || "$PREC" == 'U8_I8_I32_F32' || "$PREC" == 'I8_U8_I32_F32' || "$PREC" == 'BF8_BF8_F32_F32' || "$PREC" == 'BF8_BF8_F32_BF8' || "$PREC" == 'HF8_HF8_F32_F32' || "$PREC" == 'HF8_HF8_F32_HF8' || "$PREC" == 'F16_F16_IMPLICIT_F16' || "$PREC" == 'I8_F16_IMPLICIT_F16' || "$PREC" == 'F16_F16_F16_F16' || "$PREC" == 'I8_F16_F16_F16' || "$PREC" == 'F16_F16_F32_F16' || "$PREC" == 'I8_F16_F32_F16' || "$PREC" == 'F16_F16_IMPLICIT_F32' || "$PREC" == 'I8_F16_IMPLICIT_F32' || "$PREC" == 'F16_F16_F16_F32' || "$PREC" == 'I8_F16_F16_F32' || "$PREC" == 'F16_F16_F32_F32' || "$PREC" == 'I8_F16_F32_F32'|| "$PREC" == 'I8_BF16_F32_F32' || "$PREC" == 'I8_BF16_F32_BF16' || "$PREC" == 'BF8_F16_IMPLICIT_F16' || "$PREC" == 'BF8_F16_F32_F16' || "$PREC" == 'BF8_F16_F16_F16' || "$PREC" == 'BF8_F16_IMPLICIT_F32' || "$PREC" == 'BF8_F16_F16_F32' || "$PREC" == 'BF8_F16_F32_F32' || "$PREC" == 'I4_F16_IMPLICIT_F16' || "$PREC" == 'I4_F16_F32_F16' || "$PREC" == 'I4_F16_F16_F16' || "$PREC" == 'I4_F16_IMPLICIT_F32' || "$PREC" == 'I4_F16_F16_F32' || "$PREC" == 'I4_F16_F32_F32' || "$PREC" == 'U4_F16_IMPLICIT_F16' || "$PREC" == 'U4_F16_F32_F16' || "$PREC" == 'U4_F16_F16_F16' || "$PREC" == 'U4_F16_IMPLICIT_F32' || "$PREC" == 'U4_F16_F16_F32' || "$PREC" == 'U4_F16_F32_F32' || "$PREC" == 'U8_F16_F16_F16' || "$PREC" == 'U8_F16_F32_F16' || "$PREC" == 'U8_F16_IMPLICIT_F32' || "$PREC" == 'U8_F16_F16_F32' || "$PREC" == 'U8_F16_F32_F32' || "$PREC" == 'U8_BF16_F32_F32' || "$PREC" == 'U8_BF16_F32_BF16' || "$PREC" == 'U8_F16_IMPLICIT_F16' ) && ( "$BVNNI" == '1' ) ]]; then
                      continue
                    fi

                    # loading A in VNNI layout is mandatory for select precision
                    if [[ ("$PREC" == 'U8_I8_I32_I32' || "$PREC" == 'I8_U8_I32_I32' || "$PREC" == 'U8_I8_I32_F32' || "$PREC" == 'I8_U8_I32_F32') && ( "$AVNNI" == '0' ) ]]; then
                      continue
                    fi

                    # storing in VNNI layout is only supported/makes sense for select precision
                    if [[ ("$PREC" == 'F64_F64_F64_F64' || "$PREC" == 'F32_F32_F32_F32' || "$PREC" == 'BF16_BF16_F32_F32' || "$PREC" == 'BF8_BF8_F32_F32' || "$PREC" == 'HF8_HF8_F32_F32' || "$PREC" == 'I16_I16_I32_I32' || "$PREC" == 'U8_I8_I32_I32' || "$PREC" == 'I8_U8_I32_I32' || "$PREC" == 'U8_I8_I32_F32' || "$PREC" == 'I8_U8_I32_F32' || "$PREC" == 'F16_F16_IMPLICIT_F16' || "$PREC" == 'I8_F16_IMPLICIT_F16' || "$PREC" == 'F16_F16_F16_F16' || "$PREC" == 'I8_F16_F16_F16' || "$PREC" == 'I8_F16_F32_F16' || "$PREC" == 'F16_F16_IMPLICIT_F32' || "$PREC" == 'I8_F16_IMPLICIT_F32' || "$PREC" == 'F16_F16_F16_F32' || "$PREC" == 'I8_F16_F16_F32' || "$PREC" == 'F16_F16_F32_F32' || "$PREC" == 'I8_F16_F32_F32' || "$PREC" == 'I8_BF16_F32_F32' || "$PREC" == 'I8_BF16_F32_BF16' || "$PREC" == 'BF8_F16_IMPLICIT_F16' || "$PREC" == 'BF8_F16_F32_F16' || "$PREC" == 'BF8_F16_F16_F16' || "$PREC" == 'BF8_F16_IMPLICIT_F32' || "$PREC" == 'BF8_F16_F16_F32' || "$PREC" == 'BF8_F16_F32_F32' || "$PREC" == 'I4_F16_IMPLICIT_F16' || "$PREC" == 'I4_F16_F32_F16' || "$PREC" == 'I4_F16_F16_F16' || "$PREC" == 'I4_F16_IMPLICIT_F32' || "$PREC" == 'I4_F16_F16_F32' || "$PREC" == 'I4_F16_F32_F32' || "$PREC" == 'U4_F16_IMPLICIT_F16' || "$PREC" == 'U4_F16_F32_F16' || "$PREC" == 'U4_F16_F16_F16' || "$PREC" == 'U4_F16_IMPLICIT_F32' || "$PREC" == 'U4_F16_F16_F32' || "$PREC" == 'U4_F16_F32_F32' || "$PREC" == 'U8_F16_F16_F16' || "$PREC" == 'U8_F16_F32_F16' || "$PREC" == 'U8_F16_IMPLICIT_F32' || "$PREC" == 'U8_F16_F16_F32' || "$PREC" == 'U8_F16_F32_F32' || "$PREC" == 'U8_BF16_F32_F32' || "$PREC" == 'U8_BF16_F32_BF16' || "$PREC" == 'U8_F16_IMPLICIT_F16' ) && ( "$CVNNI" == '1' ) ]]; then
                      continue
                    fi

                    # low precision has no transpose support
                    if [[ ( "$PREC" == 'BF16_BF16_F32_F32' || "$PREC" == 'BF16_BF16_F32_BF16' || (("$PREC" == 'F16_F16_F32_F32' || "$PREC" == 'F16_F16_F32_F16') && ("$AVNNI" == '1' )) || "$PREC" == 'BF8_BF8_F32_F32' || "$PREC" == 'BF8_BF8_F32_BF8' || "$PREC" == 'HF8_HF8_F32_F32' || "$PREC" == 'HF8_HF8_F32_HF8' || "$PREC" == 'I16_I16_I32_I32' || "$PREC" == 'U8_I8_I32_I32' || "$PREC" == 'I8_U8_I32_I32' || "$PREC" == 'U8_I8_I32_F32' || "$PREC" == 'I8_U8_I32_F32' || "$PREC" == 'I8_BF16_F32_F32' || "$PREC" == 'I8_BF16_F32_BF16' || "$PREC" == 'U8_BF16_F32_F32' || "$PREC" == 'U8_BF16_F32_BF16' ) && ( "$TRA" == '1'  ||  "$TRB" == '1') ]]; then
                      continue
                    fi

                    # low precision has no transpose support
                    if [[ ( "$PREC" == 'F16_F16_IMPLICIT_F16' || "$PREC" == 'I8_F16_IMPLICIT_F16' || "$PREC" == 'F16_F16_F16_F16' || "$PREC" == 'I8_F16_F16_F16' || "$PREC" == 'F16_F16_F32_F16' || "$PREC" == 'I8_F16_F32_F16' || "$PREC" == 'F16_F16_IMPLICIT_F32' || "$PREC" == 'I8_F16_IMPLICIT_F32' || "$PREC" == 'F16_F16_F16_F32' || "$PREC" == 'I8_F16_F16_F32' || "$PREC" == 'F16_F16_F32_F32' || "$PREC" == 'I8_F16_F32_F32' || "$PREC" == 'BF8_F16_IMPLICIT_F16' || "$PREC" == 'BF8_F16_F32_F16' || "$PREC" == 'BF8_F16_F16_F16' || "$PREC" == 'BF8_F16_IMPLICIT_F32' || "$PREC" == 'BF8_F16_F16_F32' || "$PREC" == 'BF8_F16_F32_F32' || "$PREC" == 'I4_F16_IMPLICIT_F16' || "$PREC" == 'I4_F16_F32_F16' || "$PREC" == 'I4_F16_F16_F16' || "$PREC" == 'I4_F16_IMPLICIT_F32' || "$PREC" == 'I4_F16_F16_F32' || "$PREC" == 'I4_F16_F32_F32' || "$PREC" == 'U4_F16_IMPLICIT_F16' || "$PREC" == 'U4_F16_F32_F16' || "$PREC" == 'U4_F16_F16_F16' || "$PREC" == 'U4_F16_IMPLICIT_F32' || "$PREC" == 'U4_F16_F16_F32' || "$PREC" == 'U4_F16_F32_F32' || "$PREC" == 'U8_F16_F16_F16' || "$PREC" == 'U8_F16_F32_F16' || "$PREC" == 'U8_F16_IMPLICIT_F32' || "$PREC" == 'U8_F16_F16_F32' || "$PREC" == 'U8_F16_F32_F32' || "$PREC" == 'U8_BF16_F32_F32' || "$PREC" == 'U8_BF16_F32_BF16' || "$PREC" == 'U8_F16_IMPLICIT_F16') && ( "$TRA" == '1' ) ]]; then
                      continue
                    fi

                    # only FP8 has stack tansforms for data prep
                    if [[ ( "$PREC" != 'BF8_BF8_F32_F32' && "$PREC" != 'BF8_BF8_F32_BF8' && "$PREC" != 'HF8_HF8_F32_F32' && "$PREC" != 'HF8_HF8_F32_HF8' ) && ( "$STACK" == '1' ) ]]; then
                      continue
                    fi

                    if [ "$PREC" == 'F64_F64_F64_F64' ] ; then
                      OUTNAME="dgemm_"
                    elif [ "$PREC" == 'F32_F32_F32_F32' ] ; then
                      OUTNAME="sgemm_"
                    elif [ "$PREC" == 'I16_I16_I32_I32' ] ; then
                      OUTNAME="i16i32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'I8_BF16_F32_F32' ] ; then
                      OUTNAME="i8bf16f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U8_BF16_F32_F32' ] ; then
                      OUTNAME="u8bf16f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'I8_BF16_F32_BF16' ] ; then
                      OUTNAME="i8bf16bf16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U8_BF16_F32_BF16' ] ; then
                      OUTNAME="u8bf16bf16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'F16_F16_IMPLICIT_F16' ] ; then
                      OUTNAME="f16f16implicitf16gemm_"
                    elif [ "$PREC" == 'I8_F16_IMPLICIT_F16' ] ; then
                      OUTNAME="i8f16implicitf16gemm_"
                    elif [ "$PREC" == 'U8_F16_IMPLICIT_F16' ] ; then
                      OUTNAME="u8f16implicitf16gemm_"
                    elif [ "$PREC" == 'BF8_F16_IMPLICIT_F16' ] ; then
                      OUTNAME="bf8f16implicitf16gemm_"
                    elif [ "$PREC" == 'I4_F16_IMPLICIT_F16' ] ; then
                      OUTNAME="i4f16implicitf16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U4_F16_IMPLICIT_F16' ] ; then
                      OUTNAME="u4f16implicitf16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'F16_F16_F16_F16' ] ; then
                      OUTNAME="f16f16f16f16gemm_"
                    elif [ "$PREC" == 'I8_F16_F16_F16' ] ; then
                      OUTNAME="i8f16f16f16gemm_"
                    elif [ "$PREC" == 'U8_F16_F16_F16' ] ; then
                      OUTNAME="u8f16f16f16gemm_"
                    elif [ "$PREC" == 'BF8_F16_F16_F16' ] ; then
                      OUTNAME="bf8f16f16f16gemm_"
                    elif [ "$PREC" == 'I4_F16_F16_F16' ] ; then
                      OUTNAME="i4f16f16f16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U4_F16_F16_F16' ] ; then
                      OUTNAME="u4f16f16f16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'F16_F16_F32_F16') && ("$AVNNI" == '1')  ]] ; then
                      OUTNAME="f16f16f32f16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'F16_F16_F32_F16') && ("$AVNNI" == '0')  ]] ; then
                      OUTNAME="f16f16f32f16_flatgemm_"
                    elif [ "$PREC" == 'I8_F16_F32_F16' ] ; then
                      OUTNAME="i8f16f32f16gemm_"
                    elif [ "$PREC" == 'U8_F16_F32_F16' ] ; then
                      OUTNAME="u8f16f32f16gemm_"
                    elif [ "$PREC" == 'BF8_F16_F32_F16' ] ; then
                      OUTNAME="bf8f16f32f16gemm_"
                    elif [ "$PREC" == 'I4_F16_F32_F16' ] ; then
                      OUTNAME="i4f16f32f16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U4_F16_F32_F16' ] ; then
                      OUTNAME="u4f16f32f16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'F16_F16_IMPLICIT_F32' ] ; then
                      OUTNAME="f16f16implicitf32gemm_"
                    elif [ "$PREC" == 'I8_F16_IMPLICIT_F32' ] ; then
                      OUTNAME="i8f16implicitf32gemm_"
                    elif [ "$PREC" == 'U8_F16_IMPLICIT_F32' ] ; then
                      OUTNAME="u8f16implicitf32gemm_"
                    elif [ "$PREC" == 'BF8_F16_IMPLICIT_F32' ] ; then
                      OUTNAME="bf8f16implicitf32gemm_"
                    elif [ "$PREC" == 'I4_F16_IMPLICIT_F32' ] ; then
                      OUTNAME="i4f16implicitf32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U4_F16_IMPLICIT_F32' ] ; then
                      OUTNAME="u4f16implicitf32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'F16_F16_F16_F32' ] ; then
                      OUTNAME="f16f16f16f32gemm_"
                    elif [ "$PREC" == 'I8_F16_F16_F32' ] ; then
                      OUTNAME="i8f16f16f32gemm_"
                    elif [ "$PREC" == 'U8_F16_F16_F32' ] ; then
                      OUTNAME="u8f16f16f32gemm_"
                    elif [ "$PREC" == 'BF8_F16_F16_F32' ] ; then
                      OUTNAME="bf8f16f16f32gemm_"
                    elif [ "$PREC" == 'I4_F16_F16_F32' ] ; then
                      OUTNAME="i4f16f16f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U4_F16_F16_F32' ] ; then
                      OUTNAME="u4f16f16f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'F16_F16_F32_F32') && ("$AVNNI" == '1')  ]] ; then
                      OUTNAME="f16f16f32f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'F16_F16_F32_F32') && ("$AVNNI" == '0')  ]] ; then
                      OUTNAME="f16f16f32f32_flatgemm_"
                    elif [ "$PREC" == 'I8_F16_F32_F32' ] ; then
                      OUTNAME="i8f16f32f32gemm_"
                    elif [ "$PREC" == 'U8_F16_F32_F32' ] ; then
                      OUTNAME="u8f16f32f32gemm_"
                    elif [ "$PREC" == 'BF8_F16_F32_F32' ] ; then
                      OUTNAME="bf8f16f32f32gemm_"
                    elif [ "$PREC" == 'I4_F16_F32_F32' ] ; then
                      OUTNAME="i4f16f32f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U4_F16_F32_F32' ] ; then
                      OUTNAME="u4f16f32f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [ "$PREC" == 'U8_I8_I32_I32' ] ; then
                      OUTNAME="usi8i32gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [ "$PREC" == 'I8_U8_I32_I32' ] ; then
                      OUTNAME="sui8i32gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [ "$PREC" == 'U8_I8_I32_F32' ] ; then
                      OUTNAME="usi8f32gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [ "$PREC" == 'I8_U8_I32_F32' ] ; then
                      OUTNAME="sui8f32gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'BF16_BF16_F32_BF16') && ("$AVNNI" == '1') ]] ; then
                      OUTNAME="bf16gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'BF16_BF16_F32_F32') && ("$AVNNI" == '1') ]] ; then
                      OUTNAME="bf16f32gemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'BF16_BF16_F32_BF16') && ("$AVNNI" == '0') ]] ; then
                      OUTNAME="bf16_flatgemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'BF16_BF16_F32_F32') && ("$AVNNI" == '0') ]] ; then
                      OUTNAME="bf16f32_flatgemm_"
                      KSTART=2
                      KSTEP=2
                    elif [[ ("$PREC" == 'BF8_BF8_F32_BF8') && ("$AVNNI" == '1') ]] ; then
                      OUTNAME="bf8gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'BF8_BF8_F32_F32') && ("$AVNNI" == '1') ]] ; then
                      OUTNAME="bf8f32gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'HF8_HF8_F32_HF8') && ("$AVNNI" == '1') ]] ; then
                      OUTNAME="hf8gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'HF8_HF8_F32_F32') && ("$AVNNI" == '1') ]] ; then
                      OUTNAME="hf8f32gemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'BF8_BF8_F32_BF8') && ("$AVNNI" == '0') ]] ; then
                      OUTNAME="bf8_flatgemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'BF8_BF8_F32_F32') && ("$AVNNI" == '0') ]] ; then
                      OUTNAME="bf8f32_flatgemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'HF8_HF8_F32_HF8') && ("$AVNNI" == '0') ]] ; then
                      OUTNAME="hf8_flatgemm_"
                      KSTART=4
                      KSTEP=4
                    elif [[ ("$PREC" == 'HF8_HF8_F32_F32') && ("$AVNNI" == '0') ]] ; then
                      OUTNAME="hf8f32_flatgemm_"
                      KSTART=4
                      KSTEP=4
                     else
                      continue
                    fi

                    if [ "$CVNNI" == '1' ] ; then
                      if [ "$PREC" == 'BF16_BF16_F32_BF16' ] ; then
                        NSTART=2
                        NSTEP=2
                      elif [ "$PREC" == 'F16_F16_F32_F16' ] ; then
                        NSTART=2
                        NSTEP=2
                      elif [ "$PREC" == 'BF8_BF8_F32_BF8' ] ; then
                        NSTART=4
                        NSTEP=4
                      elif [ "$PREC" == 'HF8_HF8_F32_HF8' ] ; then
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
                      if [ "$CVNNI" == '1' ] ; then
                        OUTNAME=$OUTNAME"_bop"$BINARY_POSTOP"_uop"$UNARY_POSTOP"_cvnni"$CVNNI".slurm"
                      else
                        OUTNAME=$OUTNAME".slurm"
                      fi
                    fi

                    #echo "Copying "$TPLNAME" to "$OUTNAME
                    sed "s/PREC=0/PREC=\"${PREC}\"/g" ${HERE}/gemm_kernel_fused.tpl \
                    | sed "s/TRA=0/TRA=${TRA}/g" \
                    | sed "s/TRB=0/TRB=${TRB}/g" \
                    | sed "s/BINARY_POSTOP=0/BINARY_POSTOP=${BINARY_POSTOP}/g" \
                    | sed "s/UNARY_POSTOP=0/UNARY_POSTOP=${UNARY_POSTOP}/g" \
                    | sed "s/AVNNI=0/AVNNI=${AVNNI}/g" \
                    | sed "s/BVNNI=0/BVNNI=${BVNNI}/g" \
                    | sed "s/CVNNI=0/CVNNI=${CVNNI}/g" \
                    | sed "s/MSTART/${MSTART}/g" \
                    | sed "s/MSTEP/${MSTEP}/g" \
                    | sed "s/NSTART/${NSTART}/g" \
                    | sed "s/NSTEP/${NSTEP}/g" \
                    | sed "s/KSTART/${KSTART}/g" \
                    | sed "s/KSTEP/${KSTEP}/g" \
                    | sed "s/SAMPLESIZE/${SAMPLESIZE}/g" \
                    >${HERE}/${OUTNAME}

                    # for gt we need to touch up the script
                    if [ "$LD" == 'gtld' ] ; then
                      sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ '100 100 100'/g" ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    # trB we need to switch LDB
                    if [ "$TRA" == '1' ] ; then
                      sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(k) + ' ' + str(k) + ' ' + str(m)/g" ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    # trB we need to switch LDB
                    if [ "$TRB" == '1' ] ; then
                      sed "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' ' + str(n) + ' ' + str(m)/g" ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    # stack exports
                    if [[ ( "$PREC" == 'BF8_BF8_F32_BF8' || "$PREC" == 'BF8_BF8_F32_F32' ) && ( "$STACK" == '1' ) ]]; then
                      sed 's/LIBXSMM_ENV_VAR/LIBXSMM_BF8_GEMM_VIA_STACK/g' ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi
                    if [[ ( "$PREC" == 'HF8_HF8_F32_HF8' || "$PREC" == 'HF8_HF8_F32_F32' ) && ( "$STACK" == '1' ) ]]; then
                      sed 's/LIBXSMM_ENV_VAR/LIBXSMM_HF8_GEMM_VIA_STACK/g' ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    # nofusion, we use the regular kernel
                    if [ "$NOFUSION" == '1' ] ; then
                      sed 's/gemm_kernel_fused/gemm_kernel/g' ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                      sed 's/ ${BINARY_POSTOP} ${UNARY_POSTOP}//g' ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    # Export BFDOT variable
                    if [[ ( "$PREC" == 'BF16_BF16_F32_F32' || "$PREC" == 'BF16_BF16_F32_BF16' ) && ("$AVNNI" == '1') ]] ; then
                      sed 's/LIBXSMM_ENV_VAR/LIBXSMM_AARCH64_USE_BFDOT/g' ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    # Export IDOT variable
                    if [[ ( "$PREC" == 'I8_U8_I32_F32' || "$PREC" == 'I8_U8_I32_I32' ) && ("$AVNNI" == '1') ]] ; then
                      sed 's/LIBXSMM_ENV_VAR/LIBXSMM_AARCH64_USE_I8DOT/g' ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    # MMLA for BF16
                    if [[ ( "$PREC" == 'BF16_BF16_F32_F32' || "$PREC" == 'BF16_BF16_F32_BF16' ) && ("$AVNNI" == '1') ]] ; then
                      cp ${HERE}/${OUTNAME} ${HERE}/mmla_${OUTNAME}
                      sed 's/randnumk = rnd.sample(range(2,101,2)/randnumk = rnd.sample(range(4,101,4)/g' ${HERE}/mmla_${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/mmla_${OUTNAME}
                      sed 's/LIBXSMM_AARCH64_USE_BFDOT=1/LIBXSMM_AARCH64_USE_BFDOT=0/g' ${HERE}/mmla_${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/mmla_${OUTNAME}
                      if [ "$CVNNI" == '1' ] ; then
                        sed 's/randnumn = rnd.sample(range(2,101,2)/randnumn = rnd.sample(range(4,101,4)/g' ${HERE}/mmla_${OUTNAME} >${TMPFILE}
                        cp ${TMPFILE} ${HERE}/mmla_${OUTNAME}
                      fi
                      chmod 755 ${HERE}/mmla_${OUTNAME}

                      # create MMLA scripts with B in VNNIT
                      NEWNAME=mmla_bvnni_${OUTNAME}
                      NEWNAME="${NEWNAME/_nn_/_nt_}"
                      sed \
                        -e "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' ' + str(n) + ' ' + str(m)/g" \
                        -e "s/BVNNI=0/BVNNI=1/g" \
                        -e 's/TRB=0/TRB=1/g' \
                        ${HERE}/mmla_${OUTNAME} >${HERE}/${NEWNAME}
                      chmod 755 ${HERE}/${NEWNAME}
                    fi

                    # MMLA for I8
                    if [[ ( "$PREC" == 'I8_U8_I32_F32' || "$PREC" == 'I8_U8_I32_I32' ) && ("$AVNNI" == '1') ]] ; then
                      cp ${HERE}/${OUTNAME} ${HERE}/mmla_${OUTNAME}
                      sed 's/randnumk = rnd.sample(range(4,101,4)/randnumk = rnd.sample(range(8,201,8)/g' ${HERE}/mmla_${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/mmla_${OUTNAME}
                      if [ "$LD" == 'gtld' ] ; then
                        sed "s/+ '100 100 100'/+ '100 200 100'/g" ${HERE}/mmla_${OUTNAME} >${TMPFILE}
                        cp ${TMPFILE} ${HERE}/mmla_${OUTNAME}
                      fi
                      sed 's/LIBXSMM_AARCH64_USE_I8DOT=1/LIBXSMM_AARCH64_USE_I8DOT=0/g' ${HERE}/mmla_${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/mmla_${OUTNAME}
                      chmod 755 ${HERE}/mmla_${OUTNAME}
                    fi

                    # BFDOT for BF16 wth B in VNNIT
                    if [[ ( "$PREC" == 'BF16_BF16_F32_F32' || "$PREC" == 'BF16_BF16_F32_BF16' ) && ("$AVNNI" == '1') ]] ; then
                      # create BFDOT scripts with B in VNNIT
                      NEWNAME=bfdot_bvnni_${OUTNAME}
                      NEWNAME="${NEWNAME/_nn_/_nt_}"
                      sed \
                        -e "s/+ str(m) + ' ' + str(k) + ' ' + str(m)/+ str(m) + ' ' + str(n) + ' ' + str(m)/g" \
                        -e "s/BVNNI=0/BVNNI=1/g" \
                        -e 's/TRB=0/TRB=1/g' \
                        ${HERE}/${OUTNAME} >${HERE}/${NEWNAME}
                      chmod 755 ${HERE}/${NEWNAME}
                    fi

                    # remove env variable
                    if [ "$STACK" == '0' ] ; then
                      sed "/export LIBXSMM_ENV_VAR=1/d" ${HERE}/${OUTNAME} >${TMPFILE}
                      cp ${TMPFILE} ${HERE}/${OUTNAME}
                    fi

                    chmod 755 ${HERE}/${OUTNAME}

                  done
                done
              done
            done
          done
        done
      done
    done
  done
done
