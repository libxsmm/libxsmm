SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd)"
BASE_DIR=`dirname ${SCRIPT_DIR}`
PREFIX=${BASE_DIR}/miniconda3

source oneCCL/install/env/setvars.sh

# REF env
source ${PREFIX}/bin/activate ref
( cd transformers && python setup.py install )

# OPT env
source ${PREFIX}/bin/activate tpp
( cd torch-ccl && python setup.py  clean && python setup.py install )

( cd transformers && python setup.py install )

# AVX512 env
source ${PREFIX}/bin/activate avx512
( cd transformers && python setup.py install )


