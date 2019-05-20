#!/bin/bash

#SBATCH -J sgemm_nn_gtld_kernel_knm_check
#SBATCH --get-user-env
#SBATCH --partition=knm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=288
#SBATCH --time=02:00:00

LIBXSMM_TARGET=knm

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
