#!/usr/bin/env bash

TESTFILE1=$(mktemp -p .)

if [ -x "$(command -v python3)" ]; then
  PYTHON=$(command -v python3)
else
  PYTHON=$(command -v python)
fi

${PYTHON} << END
#import random as rnd
#import time as time
#rnd.seed(time.time())
#randnum = rnd.sample(range(1,101), 18)
square_range = range(2, 129, 2)
#mrange = range(2, 2, 128)
#nrange = range(2, 2, 128)
#mrange = range(2, 2, 128)
f1 = open("${TESTFILE1}", "w+")
for size in square_range:
  m = size
  n = size
  k = size
  line = str(m) + ' ' + str(n) + ' ' + str(k) + ' ' \
       + str(k) + ' ' + str(k) + ' ' + str(m) + '\n'
  f1.write(line)
#for m in randnum:
#    for n in randnum:
#        for k in randnum:
#            line = str(m) + ' ' + str(n) + ' ' + str(k) + ' ' \
#                 + str(k) + ' ' + str(k) + ' ' + str(m) + '\n'
#            f1.write(line)
f1.close()
END

echo "Non-transpose"
./kernel ${TESTFILE1} 1 1 0 0 0 0 SP nobr 1 0 1 1

echo "Transpose (gather-based or stack-based?)"
./kernel ${TESTFILE1} 1 1 0 0 1 0 SP nobr 1 0 1 1

rm ${TESTFILE1}
