#!/usr/bin/env bash

for i in `ls *.bin`
do
  echo ${i}
  /swtools/sde/kits/latest/xed64 -64 -ir ${i} | head -n 32
done
