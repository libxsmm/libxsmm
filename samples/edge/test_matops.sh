#!/bin/bash

#sde can be downloaded here
SDE64_BIN=/swtools/sde/kits/latest/sde64
SDE64_ARCH="-knl"
SDE64_FLAGS="-ptr_check -null_check -ptr_raise"
SDE=${SDE64_BIN}" "${SDE64_FLAGS}" "${SDE64_ARCH}" -- "
GREP=$(which grep 2> /dev/null)

#on an AVX512 pfatform we can run natively
CPUFLAGS=$(if [ "" != "${GREP}" ] && [ -e /proc/cpuinfo ]; then ${GREP} -m1 flags /proc/cpuinfo | cut -d: -f2-; fi)
if [ "" != "$(echo "${CPUFLAGS}" | ${GREP} -o avx512f)" ]; then
  SDE=
fi

#iterastions and order
REPS=1
PDEG=5
PREC=f32

if [[ $PDEG == "1" ]]
then
  K=4
  N=3
elif [[ $PDEG == "2" ]]
then
  K=10
  N=6
elif [[ $PDEG == "3" ]]
then
  K=20
  N=10
elif [[ $PDEG == "4" ]]
then
  K=35
  N=15
elif [[ $PDEG == "5" ]]
then
  K=56
  N=21
elif [[ $PDEG == "6" ]]
then
  K=84
  N=28
else
  echo "PDEG need to be in the range of 1 to 6"
  return -1
fi

if [[ $PREC == "f32" ]]
then
  CRUN=16
#  CRUN=8
elif [[ $PREC == "f64" ]]
then
  CRUN=8
#  CRUN=4
else
  echo "PREC needs to be either f32/f64"
  return -2
fi

M=9

# test flux matrices, CSR
for i in `ls mats/tet4_${PDEG}_fluxN*_csr.mtx`; do ${SDE} ./bsparse_srsoa_${PREC} ${M} ${N} ${K} ${CRUN} ${REPS} $i; done
for i in `ls mats/tet4_${PDEG}_fluxT*_csr.mtx`; do ${SDE} ./bsparse_srsoa_${PREC} ${M} ${K} ${N} ${CRUN} ${REPS} $i; done
# test stiffness matrices, CSR
for i in `ls mats/tet4_${PDEG}_stiff*_csr.mtx`; do ${SDE} ./bsparse_srsoa_${PREC} ${M} ${K} ${K} ${CRUN} ${REPS} $i; done
# test flux matrices, CSC
for i in `ls mats/tet4_${PDEG}_fluxN*_csc.mtx`; do ${SDE} ./bsparse_scsoa_${PREC} ${M} ${N} ${K} ${CRUN} ${REPS} $i; done
for i in `ls mats/tet4_${PDEG}_fluxT*_csc.mtx`; do ${SDE} ./bsparse_scsoa_${PREC} ${M} ${K} ${N} ${CRUN} ${REPS} $i; done
# test stiffness matrices, CSC
for i in `ls mats/tet4_${PDEG}_stiff*_csc.mtx`; do ${SDE} ./bsparse_scsoa_${PREC} ${M} ${K} ${K} ${CRUN} ${REPS} $i; done
# test star matrices
${SDE} ./asparse_srsoa_${PREC} ${M} ${K} ${M} ${CRUN} ${REPS} mats/tet4_starMatrix_csr.mtx
# test flux matrices
${SDE} ./asparse_srsoa_${PREC} ${M} ${K} ${M} ${CRUN} ${REPS} mats/tet4_fluxMatrix_csr_sp.mtx
${SDE} ./asparse_srsoa_${PREC} ${M} ${K} ${M} ${CRUN} ${REPS} mats/tet4_fluxMatrix_csr_de.mtx

