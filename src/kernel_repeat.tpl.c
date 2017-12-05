if ( i_conv_desc->unroll_kh == 0 ) {
  /* open KH loop, kj */
  libxsmm_generator_convolution_header_kh_loop(  io_generated_code, &l_loop_label_tracker,
      &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kh_loop );
}

/* unroll KH */
for ( l_kh = 0; l_kh < l_kh_trips; l_kh++) {
  if ( i_conv_desc->unroll_kw == 0 ) {
    /* open KW loop, ki */
    libxsmm_generator_convolution_header_kw_loop(  io_generated_code, &l_loop_label_tracker,
        &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kw_loop );
  }


  /* ifmInner loop, VLEN, ifm2, fully unrolled blocked by ofw_rb * ofw_rb */
  if ( (i_conv_desc->ofw_rb == 7)  && ( i_conv_desc->ofh_rb == 4 || i_conv_desc->ofh_rb == 3 ) ) {
      /* Call with 4 rows  */
      libxsmm_generator_convolution_forward_avx512_ifmloop_qfma_x_rows( io_generated_code, &l_gp_reg_mapping, &l_conv_kernel_config, i_conv_desc, l_kw_trips, i_conv_desc->ofh_rb);
  } else {
      libxsmm_generator_convolution_forward_avx512_ifmloop(  io_generated_code,
          &l_gp_reg_mapping,
          &l_conv_kernel_config,
          i_conv_desc,
          l_kw_trips );
  }

  if ( i_conv_desc->unroll_kw == 0 ) {
    /* close KW loop, ki */
    libxsmm_generator_convolution_footer_kw_loop(  io_generated_code, &l_loop_label_tracker,
        &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kw_loop, i_conv_desc->kw );
  }

  if ( !((i_conv_desc->kw == 1) && (i_conv_desc->kh == 1)) ) {
    libxsmm_x86_instruction_alu_imm(  io_generated_code,
        l_conv_kernel_config.alu_add_instruction,
        l_gp_reg_mapping.gp_reg_weight,
        i_conv_desc->weight_stride *  i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );

    if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2) == LIBXSMM_CONVOLUTION_PREFETCH_WEIGHT_L2 ) {
      libxsmm_x86_instruction_alu_imm(  io_generated_code,
          l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_weight_pf,
          i_conv_desc->weight_stride * i_conv_desc->kw * l_conv_kernel_config.l_ld_ifm_fil * i_conv_desc->fm_lp_block * l_conv_kernel_config.l_ld_ofm_fil * l_conv_kernel_config.datatype_size_wt );
    }

    libxsmm_x86_instruction_alu_imm( io_generated_code,
        l_conv_kernel_config.alu_add_instruction,
        l_gp_reg_mapping.gp_reg_input,
        i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block * l_conv_kernel_config.datatype_size_in );

    if ( (i_conv_desc->prefetch & LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1) == LIBXSMM_CONVOLUTION_PREFETCH_INPUT_L1 ) {
      libxsmm_x86_instruction_alu_imm( io_generated_code,
          l_conv_kernel_config.alu_add_instruction,
          l_gp_reg_mapping.gp_reg_input_pf,
          i_conv_desc->ifw_padded * l_conv_kernel_config.l_ld_ifm_act * i_conv_desc->fm_lp_block * l_conv_kernel_config.datatype_size_in );
    }
  }


}

if ( i_conv_desc->unroll_kh == 0 ) {
  /* close KH loop, kj */
  libxsmm_generator_convolution_footer_kh_loop(  io_generated_code, &l_loop_label_tracker,
      &l_conv_kernel_config, l_gp_reg_mapping.gp_reg_kh_loop, i_conv_desc->kh );
}

