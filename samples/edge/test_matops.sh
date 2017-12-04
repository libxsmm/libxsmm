#!/bin/bash

#sde can be downloaded here
SDE64_BIN=/swtools/sde/kits/latest/sde64
SDE64_ARCH="-knl"
SDE64_FLAGS="-ptr_check -null_check -ptr_raise"
SDE=${SDE64_BIN}" "${SDE64_FLAGS}" "${SDE64_ARCH}" -- "

#iterastions and order
REPS=1
ORDER=4
MODES=35
PREC=f64

# test flux matrices, CSR
for i in `ls mats/f*3D_${ORDER}_csr.mtx`; do ${SDE} ./bsparse_srsoa_${PREC} ${MODES} ${REPS} $i; done
# test stiffness matrices, CSR
for i in `ls mats/k*3D_${ORDER}_csr.mtx`; do ${SDE} ./bsparse_srsoa_${PREC} ${MODES} ${REPS} $i; done
# test flux matrices, CSC
for i in `ls mats/f*3D_${ORDER}_csc.mtx`; do ${SDE} ./bsparse_scsoa_${PREC} ${MODES} ${REPS} $i; done
# test stiffness matrices, CSC
for i in `ls mats/k*3D_${ORDER}_csc.mtx`; do ${SDE} ./bsparse_scsoa_${PREC} ${MODES} ${REPS} $i; done
# test star matrices
${SDE} ./asparse_srsoa_${PREC} ${MODES} ${REPS} mats/starMatrix_3D_csr.mtx
# test flux matrices
${SDE} ./asparse_srsoa_${PREC} ${MODES} ${REPS} mats/fluxMatrix_3D_csr_sp.mtx
${SDE} ./asparse_srsoa_${PREC} ${MODES} ${REPS} mats/fluxMatrix_3D_csr_de.mtx

