#!/bin/bash


TESTFILE=`mktemp -p .`

python << END
f = open( "$TESTFILE", "w+" )
for m in range(2, 101, 2):
    for n in range(2, 101, 2):
        for k in range(2, 101, 2):
            line = str(m) + ' ' + str(n) + ' ' + str(k) + ' 100 100 100\n'
            f.write( line )
f.close()
END

taskset -c 10 ./kernel $TESTFILE 1 1 0 0 0 0 SP 1 1

rm $TESTFILE
