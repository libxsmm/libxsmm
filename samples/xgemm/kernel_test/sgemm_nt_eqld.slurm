#!/bin/bash

TESTFILE1=$(mktemp -p .)
TESTFILE2=$(mktemp -p .)

python << END
f1 = open("${TESTFILE1}", "w+")
f2 = open("${TESTFILE2}", "w+")
i = 0
for m in range(2, 101, 2):
    for n in range(2, 101, 2):
        for k in range(2, 101, 2):
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

taskset -c 10 ./kernel ${TESTFILE1} 1 1 0 0 0 1 SP 1 1
taskset -c 10 ./kernel ${TESTFILE2} 1 1 0 0 0 1 SP 1 1

rm ${TESTFILE1}
rm ${TESTFILE2}
