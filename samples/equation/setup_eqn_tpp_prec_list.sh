#!/usr/bin/env bash

if [[ $1 == "equation_simple" ]]; then
  export EQN_PREC_LIST="0";
  if [[ ${LIBXSMM_TARGET} == "hsw" ]]; then
    export EQN_PREC_LIST="0 1 2 3 7 8 9 13";
  fi
  if [[ ${LIBXSMM_TARGET} == "neov1" ]]; then
    export EQN_PREC_LIST="0 1 2 3";
  fi
  if [[ ${LIBXSMM_TARGET} == "clx" ||  ${LIBXSMM_TARGET} == "cpx" ||  ${LIBXSMM_TARGET} == "spr" || ${LIBXSMM_TARGET} == "avx512_vl256_clx" ]]; then
    export EQN_PREC_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12 13";
  fi
fi

if [[ $1 == "equation_relu" ]]; then
  export EQN_PREC_LIST="0";
  if [[ ${LIBXSMM_TARGET} == "hsw" ]]; then
    export EQN_PREC_LIST="0 1 2 3 7 8 9";
  fi
  if [[ ${LIBXSMM_TARGET} == "neov1" ]]; then
    export EQN_PREC_LIST="0 1 2 3";
  fi
  if [[ ${LIBXSMM_TARGET} == "clx" ||  ${LIBXSMM_TARGET} == "cpx" ||  ${LIBXSMM_TARGET} == "spr" || ${LIBXSMM_TARGET} == "avx512_vl256_clx" ]]; then
    export EQN_PREC_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12";
  fi
fi

if [[ $1 == "equation_gather_reduce" ]]; then
  export EQN_PREC_LIST="0";
  if [[ ${LIBXSMM_TARGET} == "hsw" ]]; then
    export EQN_PREC_LIST="0 1 2 3 7 8 9";
  fi
  if [[ ${LIBXSMM_TARGET} == "neov1" ]]; then
    export EQN_PREC_LIST="0 1 2 3";
  fi
  if [[ ${LIBXSMM_TARGET} == "clx" ||  ${LIBXSMM_TARGET} == "cpx" ||  ${LIBXSMM_TARGET} == "spr" || ${LIBXSMM_TARGET} == "avx512_vl256_clx" ]]; then
    export EQN_PREC_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12";
  fi
fi

if [[ $1 == "equation_softmax" ]]; then
  export EQN_PREC_LIST="0";
  if [[ ${LIBXSMM_TARGET} == "hsw" ]]; then
    export EQN_PREC_LIST="0 1";
  fi
  if [[ ${LIBXSMM_TARGET} == "neov1" ]]; then
    export EQN_PREC_LIST="0 1";
  fi
  if [[ ${LIBXSMM_TARGET} == "clx" ||  ${LIBXSMM_TARGET} == "cpx" ||  ${LIBXSMM_TARGET} == "spr" || ${LIBXSMM_TARGET} == "avx512_vl256_clx" ]]; then
    export EQN_PREC_LIST="0 1";
  fi
fi

if [[ $1 == "equation_layernorm" ]]; then
  export EQN_PREC_LIST="0";
  if [[ ${LIBXSMM_TARGET} == "hsw" ]]; then
    export EQN_PREC_LIST="0 1";
  fi
  if [[ ${LIBXSMM_TARGET} == "neov1" ]]; then
    export EQN_PREC_LIST="0 1";
  fi
  if [[ ${LIBXSMM_TARGET} == "clx" ||  ${LIBXSMM_TARGET} == "cpx" ||  ${LIBXSMM_TARGET} == "spr" || ${LIBXSMM_TARGET} == "avx512_vl256_clx" ]]; then
    export EQN_PREC_LIST="0 1";
  fi
fi

if [[ $1 == "equation_split_sgd" ]]; then
  export EQN_PREC_LIST="1";
fi

if [[ ${LIBXSMM_TARGET} == "snb" ]]; then
    exit 0;
fi
if [[ ${LIBXSMM_TARGET} == "wsm" ]]; then
    exit 0;
fi

