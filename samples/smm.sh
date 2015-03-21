#!/bin/bash

env \
  KMP_AFFINITY=scatter \
  OFFLOAD_INIT=on_start \
  MIC_ENV_PREFIX=MIC \
  MIC_KMP_AFFINITY=scatter,granularity=fine \
./smm $*
