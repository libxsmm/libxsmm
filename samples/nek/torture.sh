#!/bin/sh

for i in `seq 1 1 300`
do
  ./rstr 32 32 32 32 32 32 16 > /dev/null
done
