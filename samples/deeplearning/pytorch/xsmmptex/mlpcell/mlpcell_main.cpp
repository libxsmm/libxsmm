
#include "mc_funcs.h"

#include "mlpcell_fp32.h"
#include "mlpcell_bf16.h"

void *create_handle(int N, int in_features, int out_features, int bn, int bc, int bk, bool bias, bool skip, int act, bool norm, float p, bool train, std::string dtype)
{
   // std::cout << dtype << " train: " << train << std::endl;
   if (dtype == "f32"){
       // std::cout << "Create Handle F32 "  << std::endl;
        MLPCell_F32 *handle = new MLPCell_F32(N, in_features, out_features, bn, bc, bk, bias, skip, act, norm, p, train);
        return (void*)handle;
   }
   else if(dtype == "bf16"){
       // std::cout << "Create Handle BF16 "  << std::endl;
       MLPCell_BF16 *handle = new MLPCell_BF16(N, in_features, out_features, bn, bc, bk, bias, skip, act, norm, p, train);
       return (void*)handle;
   }
   else{
       fprintf( stderr, "Can't create handle Datatype not matching...!\n");
   }

}

std::vector<at::Tensor> mlpcell_forward(void *handle_, std::vector<at::Tensor> inputs, std::string dtype)
{

    // TO-DO: Implement Skip=False and norm = True
    if (dtype == "f32"){
        // std::cout << "MF F32 "  << std::endl;
        MLPCell_F32 *handle = (MLPCell_F32*)handle_;
        RECORD_FUNCTION("mlpcell_fwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

        bool bias = handle->has_bias();
        bool skip = handle->has_skip();
        bool norm = handle->has_norm();

        if (!norm)
          return handle->fwd(inputs);
    }

    else if(dtype == "bf16"){

        // std::cout << "MF BF16 "  << std::endl;
        MLPCell_BF16 *handle = (MLPCell_BF16*)handle_;
        RECORD_FUNCTION("mlpcell_fwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

        bool bias = handle->has_bias();
        bool skip = handle->has_skip();
        bool norm = handle->has_norm();

        if(!norm)
          // printf("!skip && !norm --------------\n");
          return handle->fwd(inputs);
    }
    else{
        fprintf( stderr, "mlpcell forward function failed ...!\n");
    }
}

std::vector<at::Tensor> mlpcell_backward(void *handle_, std::vector<at::Tensor> inputs, std::string dtype)
{
   if (dtype == "f32"){
      // std::cout << "MB F32 "  << std::endl;
      MLPCell_F32 *handle = (MLPCell_F32*)handle_;
      RECORD_FUNCTION("mlpcell_bwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

      bool bias = handle->has_bias();
      bool skip = handle->has_skip();
      bool norm = handle->has_norm();
      if(!norm){
        return handle->bwd(inputs);}
    }
    else if(dtype == "bf16"){
      // std::cout << "MB BF16 "  << std::endl;
      MLPCell_BF16 *handle = (MLPCell_BF16*)handle_;
      RECORD_FUNCTION("mlpcell_bwd", std::vector<c10::IValue>({inputs[0], inputs[1]}));

      bool bias = handle->has_bias();
      bool skip = handle->has_skip();
      bool norm = handle->has_norm();
      if(!norm){
        return handle->bwd(inputs);}
      //else if(!norm){
      //  return handle->bwd_bias_skip_act_dropout(inputs);}
    }
    else{
      fprintf( stderr, "mlpcell backward function failed ...!\n");
    }
}

void destroy_handle(void* handle_, std::string dtype)
{
    if (dtype == "f32"){
        // std::cout << "DH F32 "  << std::endl;
        MLPCell_F32 *handle = (MLPCell_F32*)handle_;
        delete handle;
    }
    else if (dtype == "bf16"){
        // std::cout << "DH BF16 "  << std::endl;
        MLPCell_BF16 *handle = (MLPCell_BF16*)handle_;
        delete handle;
    }
    else{
        fprintf( stderr, "No handle to destroy ...!\n");
    }
}


std::vector<at::Tensor> dropout_forward(torch::Tensor input, float p, bool train, std::string dtype)
{
  if(!train || p == 0.0)
  {
    at::Tensor dropout_mask = at::empty(input.sizes(), torch::TensorOptions().dtype(torch::kByte));
    at::Tensor output = at::empty(input.sizes(), input.options());

    dropout_mask.zero_();
    output = input;

    return std::vector<at::Tensor>({output, dropout_mask});
  }
  else
  {
    at::Tensor output = at::empty(input.sizes(), input.options());
    int dim = input.sizes().size();
    int N = input.sizes()[0];
    int nn, nk, bn;

    if(dim == 4)
    {
      nn = input.sizes()[0];
      nk = input.sizes()[1];
      bn = input.sizes()[2];
    }

    at::Tensor dropout_mask;
    int bk, d;

    if(dim == 4)
    {
      bk = input.sizes()[3];
      d = bk % 32 == 0 ? bk/32 : bk/32 + 1;
      dropout_mask = at::empty({nn, nk, bn, d}, torch::TensorOptions().dtype(torch::kByte));
    }
    else if(dim == 2)
    {
      bk = input.sizes()[1];
      d = bk % 32 == 0 ? bk/32 : bk/32 + 1;
      dropout_mask = at::empty({N, d}, torch::TensorOptions().dtype(torch::kByte));
    }

    if (dtype == "f32")
    {

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        libxsmm_meltw_unary_param dropout_params;
        libxsmm_meltw_unary_flags dropout_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;
        int tid = omp_get_thread_num();
        int threads = omp_get_max_threads();
        int jobs, tb, te;
        if(dim == 4) {
          jobs = (nn % threads == 0) ? nn/threads : nn/threads + 1;
          tb = (tid*jobs < nn) ? tid*jobs : nn;
          te = ((tid+1)*jobs < nn) ? (tid+1)*jobs : nn;
          __mmask32 (*mask4)[nk][bn][d] = (__mmask32 (*)[nk][bn][d])(dropout_mask.data_ptr());
          float (*in)[nk][bn][bk] = (float (*)[nk][bn][bk])(input.data_ptr<float>());
          float (*out)[nk][bn][bk] = (float (*)[nk][bn][bk])(output.data_ptr<float>());

          for(int m=tb; m<te; m++) {
            for(int k=0; k<nk; k++) {
              dropout_params.in.primary = &in[m][k];
              dropout_params.in.secondary = rnd_state;
              dropout_params.in.tertiary = &p;
              dropout_params.out.primary = &out[m][k];
              dropout_params.out.secondary = &mask4[m][k];
              dropout_f32(bn, bk, &dropout_params, dropout_flags);
            }
          }
        }
        else if(dim == 2) {
          jobs = (N % threads == 0) ? N/threads : N/threads + 1;
          tb = (tid*jobs < N) ? tid*jobs : N;
          te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
          int loc_N = te - tb;
          __mmask32 (*mask2)[d] = (__mmask32 (*)[d])(dropout_mask.data_ptr());
          float (*in)[bk] = (float (*)[bk])(input.data_ptr<float>());
          float (*out)[bk] = (float (*)[bk])(output.data_ptr<float>());

          dropout_params.in.primary = &in[tb][0];
          dropout_params.in.secondary = rnd_state;
          dropout_params.in.tertiary = &p;
          dropout_params.out.primary = &out[tb][0];
          dropout_params.out.secondary = &mask2[tb][0];
          dropout_f32(loc_N, bk, &dropout_params, dropout_flags);
        }
      }

      return std::vector<at::Tensor>({output, dropout_mask});
    }
    else if (dtype == "bf16")
    {

#ifdef _OPENMP
#pragma omp parallel
#endif
      {
        libxsmm_meltw_unary_param dropout_params;
        libxsmm_meltw_unary_flags dropout_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;
        int tid = omp_get_thread_num();
        int threads = omp_get_max_threads();
        int jobs, tb, te;

        if(dim == 4) {
          jobs = (nn % threads == 0) ? nn/threads : nn/threads + 1;
          tb = (tid*jobs < nn) ? tid*jobs : nn;
          te = ((tid+1)*jobs < nn) ? (tid+1)*jobs : nn;
          __mmask32 (*mask4)[nk][bn][d] = (__mmask32 (*)[nk][bn][d])(dropout_mask.data_ptr());
          libxsmm_bfloat16 (*in)[nk][bn][bk] = (libxsmm_bfloat16 (*)[nk][bn][bk])(input.data_ptr<at::BFloat16>());
          libxsmm_bfloat16 (*out)[nk][bn][bk] = (libxsmm_bfloat16 (*)[nk][bn][bk])(output.data_ptr<at::BFloat16>());

          for(int m=tb; m<te; m++) {
            for(int k=0; k<nk; k++) {
              dropout_params.in.primary = &in[m][k];
              dropout_params.in.secondary = rnd_state;
              dropout_params.in.tertiary = &p;
              dropout_params.out.primary = &out[m][k];
              dropout_params.out.secondary = &mask4[m][k];
              dropout_bf16(bn, bk, &dropout_params, dropout_flags);
            }
          }
        }
        else if(dim == 2) {
          jobs = (N % threads == 0) ? N/threads : N/threads + 1;
          tb = (tid*jobs < N) ? tid*jobs : N;
          te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
          int loc_N = te - tb;
          __mmask32 (*mask2)[d] = (__mmask32 (*)[d])(dropout_mask.data_ptr());
          libxsmm_bfloat16 (*in)[bk] = (libxsmm_bfloat16 (*)[bk])(input.data_ptr<at::BFloat16>());
          libxsmm_bfloat16 (*out)[bk] = (libxsmm_bfloat16 (*)[bk])(output.data_ptr<at::BFloat16>());

          dropout_params.in.primary = &in[tb][0];
          dropout_params.in.secondary = rnd_state;
          dropout_params.in.tertiary = &p;
          dropout_params.out.primary = &out[tb][0];
          dropout_params.out.secondary = &mask2[tb][0];
          dropout_bf16(loc_N, bk, &dropout_params, dropout_flags);
        }
      }

      return std::vector<at::Tensor>({output, dropout_mask});
    }
    else
    {
      fprintf( stderr, "Dropout forward function failed ...!\n");
    }
  }
}

at::Tensor dropout_backward(torch::Tensor input, torch::Tensor dropout_mask, float p, std::string dtype)
{
  at::Tensor output = at::empty(input.sizes(), input.options());
  int dim = output.sizes().size();
  int N = output.sizes()[0];
  int nn, nk, bn;
  int bk, d;

  if(dim == 4)
  {
    nn = input.sizes()[0];
    nk = input.sizes()[1];
    bn = input.sizes()[2];
  }

  if(dim == 4) {
    bk = input.sizes()[3];
    d = bk % 32 == 0 ? bk/32 : bk/32 + 1;
  }
  else if(dim == 2) {
    bk = input.sizes()[1];
    d = bk % 32 == 0 ? bk/32 : bk/32 + 1;
  }

  if (dtype == "f32")
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_max_threads();

      libxsmm_meltw_unary_param dropout_params;
      libxsmm_meltw_unary_flags dropout_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

      int jobs, tb, te;
      if(dim == 4) {
        jobs = (nn % threads == 0) ? nn/threads : nn/threads + 1;
        tb = (tid*jobs < nn) ? tid*jobs : nn;
        te = ((tid+1)*jobs < nn) ? (tid+1)*jobs : nn;
        __mmask32 (*mask4)[nk][bn][d] = (__mmask32 (*)[nk][bn][d])(dropout_mask.data_ptr());
        float (*in)[nk][bn][bk] = (float (*)[nk][bn][bk])(input.data_ptr<float>());
        float (*out)[nk][bn][bk] = (float (*)[nk][bn][bk])(output.data_ptr<float>());

        for(int m=tb; m<te; m++) {
          for(int k=0; k<nk; k++) {
            dropout_params.in.primary = &in[m][k];
            dropout_params.in.secondary = &mask4[m][k];
            dropout_params.in.tertiary = &p;
            dropout_params.out.primary = &out[m][k];
            dropout_bwd_f32(bn, bk, &dropout_params, dropout_flags);
          }
        }
      }
      else if(dim == 2) {
        jobs = (N % threads == 0) ? N/threads : N/threads + 1;
        tb = (tid*jobs < N) ? tid*jobs : N;
        te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
        int loc_N = te - tb;
        __mmask32 (*mask2)[d] = (__mmask32 (*)[d])(dropout_mask.data_ptr());
        float (*in)[bk] = (float (*)[bk])(input.data_ptr<float>());
        float (*out)[bk] = (float (*)[bk])(output.data_ptr<float>());

        dropout_params.in.primary = &in[tb][0];
        dropout_params.in.secondary = &mask2[tb][0];
        dropout_params.in.tertiary = &p;
        dropout_params.out.primary = &out[tb][0];
        dropout_bwd_f32(loc_N, bk, &dropout_params, dropout_flags);
      }
    }
    return(output);
  }
  else if (dtype == "bf16")
  {
#ifdef _OPENMP
#pragma omp parallel
#endif
    {
      int tid = omp_get_thread_num();
      int threads = omp_get_max_threads();

      libxsmm_meltw_unary_param dropout_params;
      libxsmm_meltw_unary_flags dropout_flags = LIBXSMM_MELTW_FLAG_UNARY_BITMASK;

      int jobs, tb, te;

      if(dim == 4) {
        jobs = (nn % threads == 0) ? nn/threads : nn/threads + 1;
        tb = (tid*jobs < nn) ? tid*jobs : nn;
        te = ((tid+1)*jobs < nn) ? (tid+1)*jobs : nn;
        __mmask32 (*mask4)[nk][bn][d] = (__mmask32 (*)[nk][bn][d])(dropout_mask.data_ptr());
        libxsmm_bfloat16 (*in)[nk][bn][bk] = (libxsmm_bfloat16 (*)[nk][bn][bk])(input.data_ptr<at::BFloat16>());
        libxsmm_bfloat16 (*out)[nk][bn][bk] = (libxsmm_bfloat16 (*)[nk][bn][bk])(output.data_ptr<at::BFloat16>());

        for(int m=tb; m<te; m++) {
          for(int k=0; k<nk; k++) {
            dropout_params.in.primary = &in[m][k];
            dropout_params.in.secondary = &mask4[m][k];
            dropout_params.in.tertiary = &p;
            dropout_params.out.primary = &out[m][k];
            dropout_bwd_bf16(bn, bk, &dropout_params, dropout_flags);
          }
        }
      }
      else if(dim == 2) {
        jobs = (N % threads == 0) ? N/threads : N/threads + 1;
        tb = (tid*jobs < N) ? tid*jobs : N;
        te = ((tid+1)*jobs < N) ? (tid+1)*jobs : N;
        int loc_N = te - tb;
        __mmask32 (*mask2)[d] = (__mmask32 (*)[d])(dropout_mask.data_ptr());
        libxsmm_bfloat16 (*in)[bk] = (libxsmm_bfloat16 (*)[bk])(input.data_ptr<at::BFloat16>());
        libxsmm_bfloat16 (*out)[bk] = (libxsmm_bfloat16 (*)[bk])(output.data_ptr<at::BFloat16>());

        dropout_params.in.primary = &in[tb][0];
        dropout_params.in.secondary = &mask2[tb][0];
        dropout_params.in.tertiary = &p;
        dropout_params.out.primary = &out[tb][0];
        dropout_bwd_bf16(loc_N, bk, &dropout_params, dropout_flags);
      }
    }

    return(output);
  }
  else
  {
    fprintf( stderr, "Dropout backward function failed ...!\n");
  }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("init_libxsmm", &init_libxsmm, "Initialize libxsmm");
  m.def("set_rnd_seed", &set_rnd_seed, "Set seed for Rnd generator");
  m.def("mlpcell_forward", &mlpcell_forward, "GraphSAGE MLP Cell forward");
  m.def("mlpcell_backward", &mlpcell_backward, "GraphSAGE MLP backward");
  m.def("dropout_forward", &dropout_forward, "PCL Dropout forward");
  m.def("dropout_backward", &dropout_backward, "PCL Dropout backward");
  m.def("create_handle", &create_handle, "GraphSAGE MLP Create Handle");
  m.def("destroy_handle", &destroy_handle, "GraphSAGE MLP Destroy Handle");
}

