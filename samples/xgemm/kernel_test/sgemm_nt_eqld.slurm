#!/usr/bin/env bash

TESTFILE1=$(mktemp -p .)
TESTFILE2=$(mktemp -p .)

if [ -x "$(command -v python3)" ]; then
  PYTHON=$(command -v python3)
else
  PYTHON=$(command -v python)
fi

${PYTHON} << END
import random as rnd
import time as time
rnd.seed(time.time())
randnum = rnd.sample(range(1,101), 18)
f1 = open("${TESTFILE1}", "w+")
f2 = open("${TESTFILE2}", "w+")
i = 0
for m in randnum:
    for n in randnum:
        for k in randnum:
            line = str(m) + ' ' + str(n) + ' ' + str(k) + ' ' \
                 + str(m) + ' ' + str(n) + ' ' + str(m) + '\n'
            if 0 == (i % 2):
                f1.write(line)
            else:
                f2.write(line)
            i = i + 1
f1.close()
f2.close()
END

taskset -c 3 ./kernel ${TESTFILE1} 1 1 0 0 0 1 SP nobr 1 0 1 1
taskset -c 3 ./kernel ${TESTFILE2} 1 1 0 0 0 1 SP nobr 1 0 1 1

taskset -c 3 ./kernel ${TESTFILE1} 1 1 0 0 0 1 SP addrbr 5 0 1 1

taskset -c 3 ./kernel ${TESTFILE1} 1 1 0 0 0 1 SP offsbr 5 0 1 1

taskset -c 3 ./kernel ${TESTFILE1} 1 1 0 0 0 1 SP strdbr 5 0 1 1

taskset -c 3 ./kernel ${TESTFILE2} 1 1 0 0 0 1 SP addrbr 5 1 1 1

taskset -c 3 ./kernel ${TESTFILE2} 1 1 0 0 0 1 SP offsbr 5 1 1 1

taskset -c 3 ./kernel ${TESTFILE2} 1 1 0 0 0 1 SP strdbr 5 1 1 1

rm ${TESTFILE1}
rm ${TESTFILE2}
