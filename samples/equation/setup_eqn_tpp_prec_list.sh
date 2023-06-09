#!/usr/bin/env bash

if [[ $1 == "equation_simple" ]]; then
  export EQN_PREC_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12 13";
fi

if [[ $1 == "equation_relu" ]]; then
  export EQN_PREC_LIST="0 1 2 3 4 5 6 7 8 9 10 11 12";
fi

if [[ $1 == "equation_gather_reduce" ]]; then
  export EQN_PREC_LIST="0 1 4 7 10";
fi

if [[ $1 == "equation_softmax" ]]; then
  export EQN_PREC_LIST="0 1";
fi

if [[ $1 == "equation_layernorm" ]]; then
  export EQN_PREC_LIST="0 1";
fi

if [[ $1 == "equation_split_sgd" ]]; then
  export EQN_PREC_LIST="1";
fi

if [[ $1 == "equation_bf16_x3_split_f32" ]]; then
  export EQN_PREC_LIST="1";
fi
