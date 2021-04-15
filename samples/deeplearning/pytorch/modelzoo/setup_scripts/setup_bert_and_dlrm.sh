SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd)"
BASE_DIR=`dirname ${SCRIPT_DIR}`

ROOT_DIR=`pwd`

PREFIX=${ROOT_DIR}/miniconda3
if ! test -f ${PREFIX}/bin/conda ; then
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
sh ${SCRIPT_DIR}/Miniconda3-latest-Linux-x86_64.sh -b -p ${PREFIX}
fi
#source ${PREFIX}/bin/activate

${PREFIX}/bin/conda create -y -n sc21 python=3.8 && source ${PREFIX}/bin/activate sc21
conda install -y numpy ninja pyyaml mkl mkl-include setuptools cmake cffi jemalloc tqdm future pydot scikit-learn
conda install -y -c intel numpy
conda install -y -c eumetsat expect
conda install -y -c conda-forge gperftools onnx tensorboardx libunwind
${PREFIX}/bin/conda install -y pytorch==1.6.0 torchvision==0.7.0 cpuonly -c pytorch
#conda install -y pytorch==1.7.1 torchvision==0.8.2 torchaudio==0.7.2 cpuonly -c pytorch

${PREFIX}/bin/conda create -y -n ref --clone sc21
${PREFIX}/bin/conda create -y -n tpp --clone sc21
${PREFIX}/bin/conda create -y -n avx512 --clone sc21

mkdir -p packages
cd packages

if ! test -d oneCCL ; then
( git clone https://github.com/ddkalamk/oneCCL.git && cd oneCCL && git checkout working_1.6 )
fi

if ! test -d torch-ccl ; then
( git clone https://github.com/ddkalamk/torch-ccl.git && cd torch-ccl && git checkout working_1.6 )
fi

# if ! test -d libxsmm ; then
# ( git clone https://github.com/hfp/libxsmm.git && cd libxsmm && git checkout c251963ed0 )
# fi

if ! test -d transformers ; then
( git clone https://github.com/ddkalamk/transformers.git && cd transformers && git checkout sc21/pcl-v4.0.0 )
fi
