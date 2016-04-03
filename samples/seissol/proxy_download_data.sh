#!/bin/sh

WGET=$(which wget)

DATASET="LOH1_small merapi_15e5"
KINDS="bound neigh orient sides size"

for DATA in ${DATASET} ; do
  for KIND in ${KINDS} ; do
    ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/seissol/${DATA}.nc.${KIND}
  done
done

