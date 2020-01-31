#!/usr/bin/env bash
LIB_PATH=/swtools/caffe_deps/lib
export LD_LIBRARY_PATH=${LIB_PATH}:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=`pwd`/build/lib:$LD_LIBRARY_PATH
export BIN_PATH=/swtools/caffe_deps/bin
export PATH=${BIN_PATH}:$PATH

source /swtools/intel/compilers_and_libraries_2019.4.243/linux/bin/compilervars.sh intel64
#source /swtools/intel/compilers_and_libraries_2019.3.199/linux/mpi/intel64/bin/mpivars.sh
source /swtools/intel/compilers_and_libraries_2019.4.243/linux/tbb/bin/tbbvars.sh intel64
source /swtools/intel/impi/2017.3.196/bin64/mpivars.sh

export MLSL_ROOT=/nfs_home/savancha/MLSL/_install

if [ -z "${I_MPI_ROOT}" ]
then
    export I_MPI_ROOT="${MLSL_ROOT}"
fi

if [ -z "${PATH}" ]
then
    export PATH="${MLSL_ROOT}/intel64/bin"
else
    export PATH="${MLSL_ROOT}/intel64/bin:${PATH}"
fi

if [ -z "${LD_LIBRARY_PATH}" ]
then
    export LD_LIBRARY_PATH="${MLSL_ROOT}/intel64/lib"
else
    export LD_LIBRARY_PATH="${MLSL_ROOT}/intel64/lib:${LD_LIBRARY_PATH}"
fi
