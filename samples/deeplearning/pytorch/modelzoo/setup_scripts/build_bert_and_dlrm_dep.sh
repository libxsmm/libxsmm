
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd)"
BASE_DIR=`dirname ${SCRIPT_DIR}`
PREFIX=${BASE_DIR}/miniconda3
cd packages
sh ${SCRIPT_DIR}/build_xsmm_and_oneccl.sh
export LIBXSMM_ROOT=`realpath $BASE_DIR/../../../..`
source $BASE_DIR/packages/oneCCL/install/env/setvars.sh
sh ${SCRIPT_DIR}/build_xformers_and_torch_ccl.sh
cd ${BASE_DIR}/../xsmmptex
sh ${SCRIPT_DIR}/build_extensions.sh
