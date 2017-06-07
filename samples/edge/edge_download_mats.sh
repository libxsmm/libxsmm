#!/bin/sh

HERE=$(cd $(dirname $0); pwd -P)
MKDIR=$(which mkdir)
WGET=$(which wget)

DATASET="fM1DivM fM2DivM fM3DivM fM4DivM fP111DivM fP112DivM fP113DivM fP121DivM fP122DivM fP123DivM fP131DivM fP132DivM fP133DivM fP141DivM fP142DivM fP143DivM fP211DivM fP212DivM fP213DivM fP221DivM fP222DivM fP223DivM fP231DivM fP232DivM fP233DivM fP241DivM fP242DivM fP243DivM fP311DivM fP312DivM fP313DivM fP321DivM fP322DivM fP323DivM fP331DivM fP332DivM fP333DivM fP341DivM fP342DivM fP343DivM fP411DivM fP412DivM fP413DivM fP421DivM fP422DivM fP423DivM fP431DivM fP432DivM fP433DivM fP441DivM fP442DivM fP443DivM kEta kEtaDivM kEtaDivMT kXi kXiDivM kXiDivMT kZeta kZetaDivM kZetaDivMT m"
NUMS="1 2 3 4 5 6"

if [ "" != "${MKDIR}" ] && [ "" != "${WGET}" ]; then
  ${MKDIR} -p ${HERE}/mats; cd ${HERE}/mats
  ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/edge/mats/fluxMatrix_3D_csr_de.mtx
  ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/edge/mats/fluxMatrix_3D_csr_sp.mtx
  ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/edge/mats/starMatrix_3D_csr.mtx
  for DATA in ${DATASET}; do
    for NUM in ${NUMS}; do
      ${WGET} -N https://github.com/hfp/libxsmm/raw/master/samples/edge/mats/${DATA}_3D_${NUM}_csr.mtx
    done
  done
fi

