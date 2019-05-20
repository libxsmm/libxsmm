#!/bin/bash

#SBATCH -J dgemm_nn_eqld_kernel_wsm_check
#SBATCH --get-user-env
#SBATCH --partition=skx
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=112
#SBATCH --time=02:00:00

LIBXSMM_TARGET=wsm

TESTFILE=`mktemp -p .`

python << END
f = open( "$TESTFILE", "w+" )
for m in range(2, 101, 2):
    for n in range(2, 101, 2):
        for k in range(2, 101, 2):
            line = str(m) + ' ' + str(n) + ' ' + str(k) + ' ' + str(m) + ' ' + str(k) + ' ' + str(m) + '\n'
            f.write( line )
f.close()
END

taskset -c 10 ./kernel $TESTFILE 1 1 0 0 0 0 DP 1 1

rm $TESTFILE
