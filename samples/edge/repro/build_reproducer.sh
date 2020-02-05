#!/usr/bin/env sh
#


#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of Intel Corporation nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# global build group variables
if [[ -z $EDGE_CXX  ]]
then
  EDGE_CXX=icpc
fi

if [[ -z $EDGE_ARCH ]]
then
  EDGE_ARCH=avx512
fi
if [[ "${EDGE_ARCH}" == "avx512" ]]
then
  XSMM_ARCH=
elif [[ "${EDGE_ARCH}" == "skx" ]]
then
  XSMM_ARCH="MIC=0"
elif [[ "${EDGE_ARCH}" == "knl" ]]
then
  XSMM_ARCH="MIC=1"
elif [[ "${EDGE_ARCH}" == "knm" ]]
then
  XSMM_ARCH="MIC=2"
elif [[ "${EDGE_ARCH}" == "hsw" ]]
then
  XSMM_ARCH="AVX=2"
elif [[ "${EDGE_ARCH}" == "snb" ]]
then
  XSMM_ARCH="AVX=2"
else
  XSMM_ARCH=
fi

if [[ -z $EDGE_PARALLEL ]]
then
  EDGE_PARALLEL=omp
elif [[ "${EDGE_PARALLEL}" != "omp" ]]
then
  echo "uncompatible config for EDGE_PARALLEL="${EDGE_PARALLEL}
  echo "only support OpenMP (EDGE_PARALLEL=omp)"
  exit -1
fi

if [[ -z $EDGE_ELEMENT ]]
then
  EDGE_ELEMENT=tet4
elif [[ "${EDGE_ELEMENT}" != "tet4" ]]
then
  echo "uncompatible config for EDGE_ELEMENT="${EDGE_ELEMENT}
  echo "only support Tet4 (EDGE_ELEMENT=tet4)"
  exit -1
fi

if [[ -z $EDGE_EQUATION ]]
then
  EDGE_EQUATION=elastic
elif [[ "${EDGE_EQUATION}" != "elastic" ]]
then
  echo "uncompatible config for EDGE_EQUATION="${EDGE_EQUATION}
  echo "only support Elastic (EDGE_EQUATION=elastic)"
  exit -1
fi

export EDGE_CXXFLAGS=
if [[ "${EDGE_VALIDATE}" == "1" ]]
then
  export EDGE_CXXFLAGS="-DPP_REPRODUCER_VALIDATE "${EDGE_CXXFLAGS}
fi
if [[ "${EDGE_DEBUG}" == "1" ]]
then
  export EDGE_CXXFLAGS="-DPP_REPRODUCER_DEBUG "${EDGE_CXXFLAGS}
fi

if [[ -z $EDGE_BIN_DIR ]]
then
  EDGE_BIN_DIR=./bin/
fi

# configs to build, $order_$precision_$cfr
if [[ -z $EDGE_CONFIGS ]]
then
  if [[ "${EDGE_ARCH}" == "avx512" || "${EDGE_ARCH}" == "skx" || "${EDGE_ARCH}" == "knl" ]]
  then
    EDGE_CONFIGS="2_64_8 2_32_16 3_64_8 3_32_16 4_64_8 4_32_16 5_64_8 5_32_16 6_64_8 6_32_16 7_64_8 7_32_16"
  elif [[ "${EDGE_ARCH}" == "snb" || "${EDGE_ARCH}" == "hsw" ]]
  then
    EDGE_CONFIGS="2_64_4 2_32_8 3_64_4 3_32_8 4_64_4 4_32_8 5_64_4 5_32_8 6_64_4 6_32_8 7_64_4 7_32_8"
  else
    echo "unknown arch was specified!"
    exit -1
  fi
fi

# remove bin dir
mkdir -p ${EDGE_BIN_DIR}

for c in ${EDGE_CONFIGS}
do
  # extract config
  export EDGE_ORDER=`echo ${c} | awk -F"_" '{print $1}'`
  export EDGE_PRECISION=`echo ${c} | awk -F"_" '{print $2}'`
  export EDGE_CFR=`echo ${c} | awk -F"_" '{print $3}'`

  if [ "${EDGE_CFR}" -eq "1" ]
  then
    echo "do not support single forward run"
    exit -1
  fi

  make CXX=${EDGE_CXX} ${XSMM_ARCH}

done

