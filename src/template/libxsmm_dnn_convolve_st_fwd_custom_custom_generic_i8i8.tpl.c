/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas, Alexander Heinecke, Hans Pabst (Intel Corp.)
******************************************************************************/
const int ifmblock_lp = handle->ifmblock/handle->fm_lp_block;
int imgofm1ofhofw, img, ofm1, oj, oi, ii, ij;
/* computing first logical thread */
const int ltid = tid - start_thread;
/* number of tasks that could be run in parallel */
const int w_tasks = handle->ofw/handle->fwd_ofw_rb;
const int work = handle->desc.N * handle->blocksofm * handle->ofh * w_tasks;
const int work_KHW = handle->blocksofm * handle->ofh * w_tasks;
const int work_HW = handle->ofh * w_tasks;
/* compute chunk size */
const int chunksize = (work % handle->desc.threads == 0) ? (work / handle->desc.threads) : ((work / handle->desc.threads) + 1);
/* compute thr_begin and thr_end */
const int thr_begin = (ltid * chunksize < work) ? (ltid * chunksize) : work;
const int thr_end = ((ltid + 1) * chunksize < work) ? ((ltid + 1) * chunksize) : work;
/* Batch reduce related variables */
unsigned long long n_blocks = (unsigned long long)handle->blocksifm_blocking * handle->desc.R * handle->desc.S;
/* Calculate scaling factor here for output... */
float _scf = libxsmm_sexp2_i8i(-(handle->reg_filter->scf + handle->reg_input->scf - handle->reg_output->scf));
/* offset output pointer in case of physical output padding */
LIBXSMM_VLA_DECL(5, element_output_type, output, (element_output_type*)handle->reg_output->data + ((size_t)handle->desc.pad_h_out * handle->ofwp + handle->desc.pad_w_out) * handle->ofmblock, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock);
LIBXSMM_VLA_DECL(5, element_input_type, input, (element_input_type*)handle->reg_input->data, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock);
LIBXSMM_VLA_DECL(7, const element_filter_type, weight, (element_filter_type*)handle->reg_filter->data, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block);

libxsmm_barrier_init(handle->barrier, ltid);
if (handle->desc.R == 1 && handle->desc.S == 1) {  /* Strided based BRGEMM  */
  for (imgofm1ofhofw = thr_begin; imgofm1ofhofw < thr_end; ++imgofm1ofhofw) {
    img = imgofm1ofhofw / work_KHW;
    ofm1 = (imgofm1ofhofw % work_KHW)/work_HW;
    oj = ((imgofm1ofhofw % work_KHW)%work_HW)/w_tasks;
    oi = (((imgofm1ofhofw % work_KHW)%work_HW)%w_tasks)*handle->fwd_ofw_rb;
    ij = oj * handle->desc.u;
    ii = oi * handle->desc.v;
    br_gemm_kernel_strided( &LIBXSMM_VLA_ACCESS(7, weight, ofm1, 0, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                            &LIBXSMM_VLA_ACCESS(5,  input,  img, 0, ij, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5, output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks, &_scf);
  }
} else { /* Offset based BRGEMM */
 for (imgofm1ofhofw = thr_begin; imgofm1ofhofw < thr_end; ++imgofm1ofhofw) {
    img = imgofm1ofhofw / work_KHW;
    ofm1 = (imgofm1ofhofw % work_KHW)/work_HW;
    oj = ((imgofm1ofhofw % work_KHW)%work_HW)/w_tasks;
    oi = (((imgofm1ofhofw % work_KHW)%work_HW)%w_tasks)*handle->fwd_ofw_rb;
    ij = oj * handle->desc.u;
    ii = oi * handle->desc.v;
    br_gemm_kernel_offset(  &LIBXSMM_VLA_ACCESS(7, weight, ofm1, 0, 0, 0, 0, 0, 0, handle->blocksifm, handle->desc.R, handle->desc.S, ifmblock_lp, handle->ofmblock, handle->fm_lp_block),
                            &LIBXSMM_VLA_ACCESS(5,  input,  img, 0, ij, ii, 0, handle->blocksifm, handle->ifhp, handle->ifwp, handle->ifmblock),
                            &LIBXSMM_VLA_ACCESS(5 , output, img, ofm1, oj, oi, 0, handle->blocksofm, handle->ofhp, handle->ofwp, handle->ofmblock), &n_blocks, handle->A_offsets, handle->B_offsets, &_scf);
  }
}
libxsmm_barrier_wait(handle->barrier, ltid);

