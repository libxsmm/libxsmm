/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/hfp/libxsmm/                        *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Dhiraj Kalamkar (Intel Corp.)
******************************************************************************/

#ifndef _XSMM_FUNCTORS_H_
#define _XSMM_FUNCTORS_H_

#include <string>
#include <unordered_map>
#include <libxsmm.h>
#include <libxsmm_intrinsics_x86.h>
#include <immintrin.h>
#include <torch/extension.h>

#define PCL_ASSERT(cond, x...) do { if(!(cond)) { printf(x); fflush(stdout); exit(1); } } while(0)
#define DECL_VLA_PTR(type, name, dims, ptr) type (*name)dims = (type (*)dims)ptr
#define ALIGNDOWN(N, A) ((N) & ~((A)-1))
namespace pcl {
  typedef at::BFloat16 bfloat16;
  inline float upconvert_to_float(float val) { return val; }
  inline float upconvert_to_float(bfloat16 val) { return (float)val; }
  template<typename T> libxsmm_datatype XsmmDtype();
  template<> libxsmm_datatype XsmmDtype<float>() { return LIBXSMM_DATATYPE_F32; }
  template<> libxsmm_datatype XsmmDtype<bfloat16>() { return LIBXSMM_DATATYPE_BF16; }
  //template<> libxsmm_datatype XsmmDtype<c10::BFloat16>() { return LIBXSMM_DATATYPE_BF16; }
  template<> libxsmm_datatype XsmmDtype<int64_t>() { return LIBXSMM_DATATYPE_I64; }
  template<> libxsmm_datatype XsmmDtype<int32_t>() { return LIBXSMM_DATATYPE_I32; }

#ifdef __AVX512F__
  inline __m512 _mm512_loadu_ps_auto (float const* mem_addr) { return _mm512_loadu_ps(mem_addr);}
  inline __m512 _mm512_maskz_loadu_ps_auto (__mmask16 k, float const* mem_addr) { return _mm512_maskz_loadu_ps (k, mem_addr); }
  inline void _mm512_storeu_ps_auto (float* mem_addr, __m512 a) { _mm512_storeu_ps (mem_addr, a); }
  inline void _mm512_mask_storeu_ps_auto (float* mem_addr, __mmask16 k, __m512 a) { _mm512_mask_storeu_ps (mem_addr, k, a); }

  inline __m512 _mm512_convert_bf_ps(__m256i a) { return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_cvtepi16_epi32(a),16)); }
  inline __m256i _mm256_convert_ps_bf(__m512 a) { return _mm512_cvtepi32_epi16(_mm512_srai_epi32(LIBXSMM_INTRINSICS_MM512_ROUNDNE_BF16(a),16)); }

  inline __m512 _mm512_loadu_ps_auto (bfloat16 const* mem_addr) { return _mm512_convert_bf_ps(_mm256_loadu_si256((__m256i*)mem_addr));}
  inline __m512 _mm512_maskz_loadu_ps_auto (__mmask16 k, bfloat16 const* mem_addr) { return _mm512_convert_bf_ps(_mm256_maskz_loadu_epi16(k, (__m256i*)mem_addr));}
  inline void _mm512_storeu_ps_auto (bfloat16* mem_addr, __m512 a) { _mm256_storeu_si256 ((__m256i*)mem_addr, _mm256_convert_ps_bf(a)); }
  inline void _mm512_mask_storeu_ps_auto (bfloat16* mem_addr, __mmask16 k, __m512 a) { _mm256_mask_storeu_epi16 ((__m256i*)mem_addr, k, _mm256_convert_ps_bf(a)); }

  inline __m512 _mm512_split_loadu_ps(bfloat16 const* hi, bfloat16 const* lo) {
    auto yh = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)hi));
    auto yl = _mm512_cvtepu16_epi32(_mm256_loadu_si256((__m256i*)lo));
    return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
  }
  inline __m512 _mm512_maskz_split_loadu_ps(__mmask16 k, bfloat16 const* hi, bfloat16 const* lo) {
    auto yh = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)hi));
    auto yl = _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(k, (__m256i*)lo));
    return _mm512_castsi512_ps(_mm512_add_epi32(_mm512_bslli_epi128(yh, 2), yl));
  }
  inline void _mm512_split_storeu_ps(bfloat16 *hi, bfloat16 *lo, __m512 a) {
    //_mm512_storeu_ps_auto(hi, a);
    _mm256_storeu_si256((__m256i*)hi, _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
    _mm256_storeu_si256((__m256i*)lo, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
  }
  inline void _mm512_mask_split_storeu_ps(bfloat16 *hi, bfloat16 *lo, __mmask16 k, __m512 a) {
    //_mm512_mask_storeu_ps_auto(hi, k, a);
    _mm256_mask_storeu_epi16((__m256i*)hi, k, _mm512_cvtepi32_epi16(_mm512_bsrli_epi128(_mm512_castps_si512(a), 2)));
    _mm256_mask_storeu_epi16((__m256i*)lo, k, _mm512_cvtepi32_epi16(_mm512_castps_si512(a)));
  }
#endif

  class BaseTPP
  {
    public:
      void *get_kernel() {
        void *kernel = NULL;
        if (hash == "") hash = hash_str();
        auto search = kernel_cache.find(hash);
        if (search != kernel_cache.end()) kernel = search->second;
        if (kernel == NULL) {
          kernel = build_kernel();
          if (kernel == NULL) {
            fprintf(stderr, "Unable to get JIT kernel for %s\n", hash.c_str());
            exit(1);
          }
          printf("TPP: %s @ %p\n", hash.c_str(), kernel);
          kernel_cache[hash] = kernel;
        }
        return kernel;
      }
    protected:
      virtual std::string hash_str() = 0;
      virtual void *build_kernel() = 0;
      std::string hash = "";
      inline static std::unordered_map<std::string, void *>kernel_cache;
      bool initialized = false;
  };

  class UnaryTPP: public BaseTPP {
    public:
      UnaryTPP() { }
      UnaryTPP(libxsmm_blasint rows, libxsmm_blasint cols, libxsmm_blasint ldi, libxsmm_blasint ldo,
          libxsmm_datatype dt_in, libxsmm_datatype dt_out, libxsmm_datatype dt_compute,
          libxsmm_meltw_unary_flags flags, libxsmm_meltw_unary_type type) : rows(rows),
      cols(cols), ldi(ldi), ldo(ldo), dt_in(dt_in), dt_out(dt_out),
      dt_compute(dt_compute), flags(flags), type(type) {
        kernel = (libxsmm_meltwfunction_unary)get_kernel();
        initialized = true;
      }

      void operator()(void *in, void *out) {
        if (!initialized) return;
        libxsmm_meltw_unary_param unary_param;
        unary_param.in.primary  = in;
        unary_param.out.primary = out;
        kernel(&unary_param);
      }
      void operator()(void *in, void *out, void *out2) {
        if (!initialized) return;
        libxsmm_meltw_unary_param unary_param;
        unary_param.in.primary  = in;
        unary_param.out.primary = out;
        unary_param.out.secondary = out2;
        kernel(&unary_param);
      }
      void operator()(void *in, void *in2, void *in3, void *out, void *out2) {
        if (!initialized) return;
        libxsmm_meltw_unary_param unary_param;
        unary_param.in.primary  = in;
        unary_param.in.secondary  = in2;
        unary_param.in.tertiary  = in3;
        unary_param.out.primary = out;
        unary_param.out.secondary = out2;
        kernel(&unary_param);
      }
    protected:
      std::string hash_str() override {
        char hash[200];
        snprintf(hash, 200, "unary_r%d_c%d_i%d_o%d_di%d_do%d_dc%d_f%d_t%d", rows, cols, ldi, ldo, dt_in, dt_out, dt_compute, flags, type);
        return std::string(hash);
      }
      void *build_kernel() override {
        return (void*)libxsmm_dispatch_meltw_unary(cols, rows, &ldi, &ldo, dt_in, dt_compute, dt_out, flags, type);
      }

      libxsmm_blasint rows = 0;
      libxsmm_blasint cols = 0;
      libxsmm_blasint ldi;
      libxsmm_blasint ldo;
      libxsmm_datatype dt_in;
      libxsmm_datatype dt_out;
      libxsmm_datatype dt_compute;
      libxsmm_meltw_unary_flags flags;
      libxsmm_meltw_unary_type type;
      libxsmm_meltwfunction_unary kernel = NULL;
  };

  class BinaryTPP: public BaseTPP {
    public:
      BinaryTPP() { }
      BinaryTPP(libxsmm_blasint rows, libxsmm_blasint cols, libxsmm_blasint ldi, libxsmm_blasint ldo,
          libxsmm_datatype dt_in, libxsmm_datatype dt_out, libxsmm_datatype dt_compute,
          libxsmm_meltw_binary_flags flags, libxsmm_meltw_binary_type type) : rows(rows),
      cols(cols), ldi(ldi), ldo(ldo), dt_in(dt_in), dt_out(dt_out),
      dt_compute(dt_compute), flags(flags), type(type) {
        kernel = (libxsmm_meltwfunction_binary)get_kernel();
        initialized = true;
      }

      void operator()(void *in0, void *in1, void *out) {
        if (!initialized) return;
        libxsmm_meltw_binary_param binary_param;
        binary_param.in0.primary  = in0;
        binary_param.in1.primary  = in1;
        binary_param.out.primary = out;
        kernel(&binary_param);
      }
    protected:
      std::string hash_str() override {
        char hash[200];
        snprintf(hash, 200, "binary_r%d_c%d_i%d_o%d_di%d_do%d_dc%d_f%d_t%d", rows, cols, ldi, ldo, dt_in, dt_out, dt_compute, flags, type);
        return std::string(hash);
      }
      void *build_kernel() override {
        return (void*)libxsmm_dispatch_meltw_binary(cols, rows, &ldi, &ldo, dt_in, dt_compute, dt_out, flags, type);
      }

      libxsmm_blasint rows = 0;
      libxsmm_blasint cols = 0;
      libxsmm_blasint ldi;
      libxsmm_blasint ldo;
      libxsmm_datatype dt_in;
      libxsmm_datatype dt_out;
      libxsmm_datatype dt_compute;
      libxsmm_meltw_binary_flags flags;
      libxsmm_meltw_binary_type type;
      libxsmm_meltwfunction_binary kernel = NULL;
  };

  template<typename T>
    class SetZeroTPP {
      public:
        SetZeroTPP() { }
        SetZeroTPP(int N) : N(N), kernel(1, N, N, N, XsmmDtype<T>(), XsmmDtype<T>(), XsmmDtype<T>(), LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_XOR) { }
        void operator()(T *buf) {
          kernel((void*)buf, (void*)buf);
        }
        void ref(T *buf) {
          for (int i = 0; i < N; i++) {
            buf[i] = 0;
          }
        }

      private:
        int N = 0;
        UnaryTPP kernel;
    };

  template<typename Tin, typename Tout>
    class ConvertTPP {
      public:
        ConvertTPP() { }
        ConvertTPP(int N) : ConvertTPP(1, N) { }
        ConvertTPP(int rows, int cols) : rows(rows), cols(cols), kernel(1, rows*cols, rows*cols, rows*cols, XsmmDtype<Tin>(), XsmmDtype<Tout>(), XsmmDtype<Tin>() == XsmmDtype<Tout>() ? XsmmDtype<Tout>() : LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY), init_done(true) { }
        void operator()(Tin *in, Tout *out) {
          kernel((void*)in, (void*)out);
        }
        void ref(Tin *in, Tout *out) {
          for (int i = 0; i < rows * cols; i++) {
            out[i] = (Tout)in[i];
          }
        }
        bool initialized() { return init_done; }
      private:
        int rows = 0;
        int cols = 0;
        UnaryTPP kernel;
        bool init_done = false;
    };

  template<typename T>
    class CpyTPP {
      public:
        CpyTPP() { }
        CpyTPP(int N) : CpyTPP(1, N) { }
        CpyTPP(int rows, int cols) : rows(rows), cols(cols), kernel(rows, cols, cols, cols, XsmmDtype<T>(), XsmmDtype<T>(), XsmmDtype<T>(), LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) { }
        void operator()(T *in, T *out) {
          kernel((void*)in, (void*)out);
        }
        void ref(T *in, T *out) {
          for (int i = 0; i < rows * cols; i++) {
            out[i] = in[i];
          }
        }
      private:
        int rows = 0;
        int cols = 0;
        UnaryTPP kernel;
    };

  template<typename T>
    class CpyBiasTPP {
      public:
        CpyBiasTPP() { }
        CpyBiasTPP(int rows, int cols) : rows(rows), cols(cols), kernel(rows, cols, cols, cols, XsmmDtype<T>(), XsmmDtype<T>(), XsmmDtype<T>(), LIBXSMM_MELTW_FLAG_UNARY_BCAST_COL, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY) { }
        void operator()(T *in, T *out) {
          kernel((void*)in, (void*)out);
        }
        void ref(T *in, T *out) {
          for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
              out[r*cols+c] = in[c];
            }
          }
        }
      private:
        int rows = 0;
        int cols = 0;
        UnaryTPP kernel;
    };

  template<typename T>
    class AddBiasTPP {
      public:
        AddBiasTPP() { }
        AddBiasTPP(int rows, int cols) : rows(rows), cols(cols),
        kernel(rows, cols, cols, cols, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0, LIBXSMM_MELTW_TYPE_BINARY_ADD),
        cvt() {
          if (!std::is_same<T,float>::value)
            cvt = ConvertTPP<T, float>(1, cols);
        }
        void operator()(T *in, float *out) {
          if (std::is_same<T, float>::value) {
            kernel((void*)in, (void*)out, (void*)out);
          } else {
            float tmp[cols];
            cvt(in, tmp);
            kernel((void*)tmp, (void*)out, (void*)out);
          }
        }
        void ref(T *in, float *out) {
          for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
              out[r*cols+c] += (float)in[c];
            }
          }
        }
      private:
        int rows = 0;
        int cols = 0;
        BinaryTPP kernel;
        ConvertTPP<T, float> cvt;
    };

  template<typename Tin>
    class GradBiasTPP {
      public:
        GradBiasTPP() { }
        GradBiasTPP(int rows, int cols) : rows(rows), cols(cols), kernel(rows, cols, cols, cols, XsmmDtype<Tin>(), LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD) { }
        void operator()(Tin *in, float *out) {
          kernel((void*)in, (void*)out);
        }
        void ref(Tin *in, float *out) {
          for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
              out[c] += (float)in[r*cols+c];
            }
          }
        }
      private:
        int rows = 0;
        int cols = 0;
        UnaryTPP kernel;
    };

  template<typename Tin, typename Tout>
    class AddTPP {
      public:
        AddTPP() { }
        AddTPP(int N) : AddTPP(1, N) { }
        AddTPP(int rows, int cols) : rows(rows), cols(cols), kernel(rows, cols, cols, cols, XsmmDtype<Tin>(), XsmmDtype<Tout>(), LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_MELTW_TYPE_BINARY_ADD) { }
        void operator()(Tin *in0, Tin *in1, Tout *out) {
          kernel((void*)in0, (void*)in1, (void*)out);
        }
        void ref(Tin *in0, Tin *in1, Tout *out) {
          for (int r = 0; r < rows; r++) {
            for (int c = 0; c < cols; c++) {
              out[r*cols+c] += (float)in0[r*cols+c] + (float)in1[r*cols+c];
            }
          }
        }
      private:
        int rows = 0;
        int cols = 0;
        BinaryTPP kernel;
    };

  template<typename Tin, typename Tout>
    class ScaleTPP {
      public:
        ScaleTPP() { }
        ScaleTPP(int N) : N(N), kernel(1, N, N, N, XsmmDtype<Tin>(), XsmmDtype<Tout>(), LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0, LIBXSMM_MELTW_TYPE_BINARY_MUL) { }
        void operator()(Tin *in, Tout *out, float scale) {
          Tin alpha = scale;
          kernel((void*)&alpha, (void*)in, (void*)out);
        }
        void ref(Tin *in, Tout *out, float scale) {
          Tin alpha = scale;
          for (int i = 0; i < N; i++) {
            out[i] = (float)in[i] * (float)alpha;
          }
        }
      private:
        int N = 0;
        BinaryTPP kernel;
    };

  template<typename T>
    class Norm2TPP {
      public:
        Norm2TPP() { }
        Norm2TPP(int N) : N(N), kernel(1, N, N, N, XsmmDtype<T>(), LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X2_OP_ADD) { }
        void operator()(T *in, float *sum) {
          float lsum =0.0f;
          kernel((void*)in, (void*)&lsum);
          *sum += lsum;
        }
        void ref(T *in, float *sum) {
          float lsum = 0.0f;
          for (int i = 0; i < N; i++) {
            lsum += (float)in[i] * (float)in[i];
          }
          *sum += lsum;
        }
      private:
        int N = 0;
        UnaryTPP kernel;
    };

  template<typename Tin, typename Tout>
    class ScaleAddTPP {
      public:
        ScaleAddTPP() { }
        ScaleAddTPP(int N) : N(N), kernel(1, N, N, N, XsmmDtype<Tin>(), XsmmDtype<Tout>(), LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_0, LIBXSMM_MELTW_TYPE_BINARY_MULADD) { }
        void operator()(Tin *in, Tout *out, float scale) {
          Tin alpha = scale;
          kernel((void*)&alpha, (void*)in, (void*)out);
        }
        void ref(Tin *in, Tout *out, float scale) {
          Tin alpha = scale;
          for (int i = 0; i < N; i++) {
            out[i] += (float)in[i] * (float)alpha;
          }
        }
      private:
        int N = 0;
        BinaryTPP kernel;
    };

  class XformTPP {
    public:
      XformTPP() { }
      XformTPP(libxsmm_blasint rows_i, libxsmm_blasint cols_i, libxsmm_blasint ldi, libxsmm_blasint ldo, libxsmm_datatype dtype, libxsmm_meltw_unary_type type) : rows(rows_i), cols(cols_i), ldi(ldi), ldo(ldo), dtype(dtype), type(type), kernel(rows, cols, ldi, ldo, dtype, dtype, dtype, LIBXSMM_MELTW_FLAG_UNARY_NONE, type) { }
      void operator()(void *in, void *out) {
        kernel(in, out);
      }
      typedef enum XFORM_TYPE {
        XFORM_NONE_TPP = 0,
        XFORM_XPOSE_TPP = 1,
        XFORM_N2V_TPP = 2,
        XFORM_XPOSE_N2V_TPP = 3,
        XFORM_XPOSE_V2V_TPP = 4
      } XFORM_TYPE;
    private:
      libxsmm_blasint rows = 0;
      libxsmm_blasint cols = 0;
      libxsmm_blasint ldi;
      libxsmm_blasint ldo;
      libxsmm_datatype dtype;
      libxsmm_meltw_unary_type type;
      UnaryTPP kernel;
  };

  template<typename T>
    class XformExtTPP {
      public:
        XformExtTPP() { }
        XformExtTPP(int rows, int cols, XformTPP::XFORM_TYPE xtype, bool is_input_rc = true) : rows(rows), cols(cols), xtype(xtype), dtype(XsmmDtype<T>()), kernel(), cvt() {
          if (is_input_rc == false && (xtype == XformTPP::XFORM_XPOSE_TPP || xtype == XformTPP::XFORM_XPOSE_N2V_TPP || xtype == XformTPP::XFORM_XPOSE_V2V_TPP)) {
            auto tmp = rows;
            rows = cols;
            cols = tmp;
          }
          if (xtype == XformTPP::XFORM_XPOSE_TPP) {
            kernel = XformTPP(rows, cols, cols, rows, dtype, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
          } else if (xtype == XformTPP::XFORM_N2V_TPP) {
            if (dtype == LIBXSMM_DATATYPE_BF16) {
              PCL_ASSERT(rows % 2 == 0, "N2VTPP: uneven number of rows\n");
              kernel = XformTPP(rows, cols, cols, cols, dtype, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_VNNI);
            } else {
              kernel = XformTPP(rows, cols, cols, cols, dtype, LIBXSMM_MELTW_TYPE_UNARY_IDENTITY);
            }
          } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
            if (dtype == LIBXSMM_DATATYPE_BF16) {
              PCL_ASSERT(cols % 2 == 0, "XposeN2VTPP: uneven number of cols\n");
              kernel = XformTPP(rows, cols/2, cols/2, rows, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
            } else {
              kernel = XformTPP(rows, cols, cols, rows, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
            }
          } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
            if (dtype == LIBXSMM_DATATYPE_BF16) {
              PCL_ASSERT(rows % 2 == 0, "XposeV2VTPP: uneven number of rows\n");
              PCL_ASSERT(cols % 2 == 0, "XposeV2VTPP: uneven number of cols\n");
              kernel = XformTPP(rows, cols, cols, rows, dtype, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_VNNI_TO_VNNIT);
            } else {
              kernel = XformTPP(rows, cols, cols, rows, dtype, LIBXSMM_MELTW_TYPE_UNARY_TRANSFORM_NORM_TO_NORMT);
            }
          }
          if (std::is_same<T, bfloat16>::value)
            cvt = ConvertTPP<float,bfloat16>(rows, cols);
        }
        void operator()(T *in, T *out) {
          kernel((void*)in, (void*)out);
        }
        void ref(T *in, T *out) {
          auto BS = dtype == LIBXSMM_DATATYPE_BF16 ? 2 : 1;
          if (xtype == XformTPP::XFORM_XPOSE_TPP) {
            for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols; j++) {
                out[j*rows+i] = in[i*cols+j];
              }
            }
          } else if (xtype == XformTPP::XFORM_N2V_TPP) {
            for (int i = 0; i < rows/BS; i++) {
              for (int j = 0; j < cols; j++) {
                for (int k = 0; k < BS; k++) {
                  out[i*cols*BS+j*BS+k] = in[i*cols*BS+k*cols+j];
                }
              }
            }
          } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
            for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols/BS; j++) {
                for (int k = 0; k < BS; k++) {
                  out[j*rows*BS+i*BS+k] = in[i*cols+j*BS+k];
                }
              }
            }
          } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
            for (int i = 0; i < rows/BS; i++) {
              for (int j = 0; j < cols/BS; j++) {
                for (int k = 0; k < BS; k++) { //RBS
                  for (int l = 0; l < BS; l++) { //CBS
                    out[j*rows*BS+i*BS*BS+k*BS+l] = in[i*cols*BS+j*BS*BS+l*BS+k];
                  }
                }
              }
            }
          } else {
            PCL_ASSERT(false, "Should not come here\n");
          }
        }
        void operator()(float *in, bfloat16 *out) {
          bfloat16 tmp[rows*cols];
          cvt(in, tmp);
          kernel((void*)tmp, (void*)out);
        }
        void ref(float *in, bfloat16 *out) {
          auto BS = dtype == LIBXSMM_DATATYPE_BF16 ? 2 : 1;
          if (xtype == XformTPP::XFORM_XPOSE_TPP) {
            for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols; j++) {
                out[j*rows+i] = in[i*cols+j];
              }
            }
          } else if (xtype == XformTPP::XFORM_N2V_TPP) {
            for (int i = 0; i < rows/BS; i++) {
              for (int j = 0; j < cols; j++) {
                for (int k = 0; k < BS; k++) {
                  out[i*cols*BS+j*BS+k] = in[i*cols*BS+k*cols+j];
                }
              }
            }
          } else if (xtype == XformTPP::XFORM_XPOSE_N2V_TPP) {
            for (int i = 0; i < rows; i++) {
              for (int j = 0; j < cols/BS; j++) {
                for (int k = 0; k < BS; k++) {
                  out[j*rows*BS+i*BS+k] = in[i*cols+j*BS+k];
                }
              }
            }
          } else if (xtype == XformTPP::XFORM_XPOSE_V2V_TPP) {
            for (int i = 0; i < rows/BS; i++) {
              for (int j = 0; j < cols/BS; j++) {
                for (int k = 0; k < BS; k++) { //RBS
                  for (int l = 0; l < BS; l++) { //CBS
                    out[j*rows*BS+i*BS*BS+k*BS+l] = in[i*cols*BS+j*BS*BS+l*BS+k];
                  }
                }
              }
            }
          } else {
            PCL_ASSERT(false, "Should not come here\n");
          }
        }
        void operator()(int count, long str_in, long str_out, T *in, T *out) {
          for(int i = 0; i < count; i++) {
            this->operator()(&in[i*str_in], &out[i*str_out]);
          }
        }
        void ref(int count, long str_in, long str_out, T *in, T *out) {
          for(int i = 0; i < count; i++) {
            this->ref(&in[i*str_in], &out[i*str_out]);
          }
        }
        void operator()(int count, long str_in, long str_out, float *in, bfloat16 *out) {
          for(int i = 0; i < count; i++) {
            this->operator()(&in[i*str_in], &out[i*str_out]);
          }
        }
        void ref(int count, long str_in, long str_out, float *in, bfloat16 *out) {
          for(int i = 0; i < count; i++) {
            this->ref(&in[i*str_in], &out[i*str_out]);
          }
        }

      private:
        libxsmm_blasint rows = 0;
        libxsmm_blasint cols = 0;
        XformTPP::XFORM_TYPE xtype;
        libxsmm_datatype dtype;
        XformTPP kernel;
        ConvertTPP<float,bfloat16> cvt;
    };

  template<typename Tin, typename Tout>
    class BrgemmTPP : public BaseTPP {
      public:
        BrgemmTPP() { }
        BrgemmTPP(long M, long N, long K, long str_a, long str_b, float beta = 1.0, int a_trans=0) : M(M), N(N), K(K), str_a(str_a), str_b(str_b), beta(beta), a_trans(a_trans) {
          auto dt_in = XsmmDtype<Tin>();
          auto dt_out = XsmmDtype<Tout>();
          long type = -1;
          if (dt_in == LIBXSMM_DATATYPE_F32) {
            PCL_ASSERT(dt_out == LIBXSMM_DATATYPE_F32, "BRGEMM Assert\n");
            type = 0;
          } else if (dt_out == LIBXSMM_DATATYPE_F32) {
            type = 1;
          } else {
            type = 2;
          }
          if (type != 0) PCL_ASSERT(a_trans == 0, "A Transpose supported only for FP32 BRGEMM\n");
          brgemm_type = type;
          kernel.smrs= (libxsmm_smmfunction_reducebatch_strd)get_kernel();
          initialized = true;
        }
        void operator()(Tin *A, Tin *B, Tout *C, unsigned long long count) {
          if (!initialized) return;
          if (brgemm_type == 0) {
            kernel.smrs((float*)B, (float*)A, (float*)C, &count);
          } else if (brgemm_type == 1) {
            kernel.bsmrs((libxsmm_bfloat16*)B, (libxsmm_bfloat16*)A, (float*)C, &count);
          } else {
            kernel.bmrs((libxsmm_bfloat16*)B, (libxsmm_bfloat16*)A, (libxsmm_bfloat16*)C, &count);
          }
        }
        void ref(Tin *A, Tin *B, Tout *C, unsigned long long count) {
          for (uint64_t c = 0; c < count; c++) {
            auto A_ = &A[c*str_a];
            auto B_ = &B[c*str_b];
            if (brgemm_type == 0) {
              for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                  if (beta == 0.0 && c == 0) C[i*N+j] = 0.0;
                  for (int k = 0; k < K; k++) {
                    if (a_trans == 1) {
                      C[i*N+j] += A_[k*M+i] * B_[k*N+j];
                    } else {
                      C[i*N+j] += A_[i*K+k] * B_[k*N+j];
                    }
                  }
                }
              }
            } else {
              for (int i = 0; i < M; i++) {
                for (int j = 0; j < N; j++) {
                  float sum = 0.0f;
                  if (beta == 1.0 && c == 0) sum = C[i*N+j];
                  for (int k = 0; k < K/2; k++) {
                    sum += (float)A_[i*K+k*2+0] * (float)B_[k*N*2+j*2+0];
                    sum += (float)A_[i*K+k*2+1] * (float)B_[k*N*2+j*2+1];
                  }
                  C[i*N+j] = (Tout)sum;
                }
              }
            }
          }
        }
      protected:
        std::string hash_str() override {
          char hash[200];
          snprintf(hash, 200, "brgemm_m%ld_n%ld_k%ld_a%ld_b%ld_t%ld_beta%d_at%d", M, N, K, str_a, str_b, brgemm_type, (int)beta, a_trans);
          return std::string(hash);
        }
        void *build_kernel() override {
          float alpha = 1.0;
          if (brgemm_type == 0) {
            int flags = LIBXSMM_GEMM_FLAGS( 'N', 'N' );
            if (a_trans == 1) flags = LIBXSMM_GEMM_FLAGS( 'N', 'T' );
            libxsmm_smmfunction_reducebatch_strd kernel = libxsmm_smmdispatch_reducebatch_strd(N, M, K, str_b*sizeof(float), str_a*sizeof(float), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
            return (void*)kernel;
          } else if (brgemm_type == 1) {
            int flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
            libxsmm_bsmmfunction_reducebatch_strd kernel = libxsmm_bsmmdispatch_reducebatch_strd(N, M, K, str_b*sizeof(bfloat16), str_a*sizeof(bfloat16), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
            return (void*)kernel;
          } else {
            int flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
            libxsmm_bmmfunction_reducebatch_strd kernel = libxsmm_bmmdispatch_reducebatch_strd(N, M, K, str_b*sizeof(bfloat16), str_a*sizeof(bfloat16), NULL, NULL, NULL, &alpha, &beta, &flags, NULL);
            return (void*)kernel;
          }
        }
      private:
        long M, N, K, str_a, str_b;
        float beta;
        int a_trans;
        libxsmm_xmmfunction kernel;
        long brgemm_type = -1;
    };

  template<typename Tin, typename Tout>
    class BrgemmExtTPP {
      public:
        BrgemmExtTPP() { }
        BrgemmExtTPP(long M, long N, long K, long str_a, long str_b, float beta = 1.0, XformTPP::XFORM_TYPE c_trans = XformTPP::XFORM_NONE_TPP, int a_trans=0) : M(M), N(N), K(K), beta(beta), c_trans(c_trans), brgemm(), xform(), add() {
          //auto dt_in = XsmmDtype<Tin>();
          auto dt_out = XsmmDtype<Tout>();
          if (dt_out == LIBXSMM_DATATYPE_F32 && c_trans == XformTPP::XFORM_N2V_TPP) c_trans = XformTPP::XFORM_NONE_TPP;
          auto beta_ = beta;

          if (c_trans != XformTPP::XFORM_NONE_TPP) {
            beta_ = 0.0;
            xform = XformExtTPP<Tout>(M, N, c_trans);
          }
          brgemm = BrgemmTPP<Tin, Tout>(M, N, K, str_a, str_b, beta_, a_trans);
          if (beta_ != beta) {
            add = AddTPP<Tout, Tout>(M, N);
          }
        }

        void operator()(Tin *A, Tin *B, Tout *C, long count) {
          if (c_trans == XformTPP::XFORM_NONE_TPP) {
            brgemm(A, B, C, count);
          } else {
            Tout tmp_C[M*N];
            brgemm(A, B, tmp_C, count);
            if (beta == 0.0) {
              xform(tmp_C, C);
            } else {
              Tout tmp[M*N];
              xform(tmp_C, tmp);
              add(C, tmp, C);
            }
          }
        }

        void ref(Tin *A, Tin *B, Tout *C, long count) {
          if (c_trans == XformTPP::XFORM_NONE_TPP) {
            brgemm.ref(A, B, C, count);
          } else {
            Tout tmp_C[M*N];
            brgemm.ref(A, B, tmp_C, count);
            if (beta == 0.0) {
              xform.ref(tmp_C, C);
            } else {
              Tout tmp[M*N];
              xform.ref(tmp_C, tmp);
              add.ref(C, tmp, C);
            }
          }
        }

      private:
        long M, N, K;
        float beta;
        XformTPP::XFORM_TYPE c_trans;
        BrgemmTPP<Tin, Tout> brgemm;
        XformExtTPP<Tout> xform;
        AddTPP<Tout, Tout> add;
    };

  template<typename T>
    class GeluFwdTPP {
      public:
        GeluFwdTPP() { }
        GeluFwdTPP(int N) : N(N), kernel(1, N, N, N, XsmmDtype<T>(), XsmmDtype<T>(), LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_GELU) { }
        void operator()(T *in, T *out) {
          kernel((void*)in, (void*)out);
        }
        void ref(T *in, T *out) {
#ifdef __AVX512F__
          int i;
          for(i = 0; i < ALIGNDOWN(N, 16); i+=16) {
            auto vin = _mm512_loadu_ps_auto(&in[i]);
            //auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
            auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
            _mm512_storeu_ps_auto(&out[i], vout);
          }
          if(i < N) {
            int rem = N - i;
            __mmask16 mask = (1 << rem) - 1;
            auto vin = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
            //auto vout = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_FWD(vin);
            auto vout = LIBXSMM_INTRINSICS_MM512_GELU_FWD_PS_MINIMAX3(vin);
            _mm512_mask_storeu_ps_auto(&out[i], mask, vout);
          }
#else
          for (int i = 0; i < N; i++) {
            float x = in[i];
            out[i] = (erff(x/sqrtf(2.0)) + 1.0)*0.5*x;
          }
#endif
        }
      private:
        int N = 0;
        UnaryTPP kernel;
    };

  template<typename T>
    class GeluBwdTPP : public BaseTPP {
      public:
        GeluBwdTPP() { }
        GeluBwdTPP(int N) : N(N){
          kernel = (libxsmm_matrix_eqn_function)get_kernel();
          initialized = true;
        }
        void operator()(T *gout, T *in, T *gin) {
          if (!initialized) return;
          libxsmm_matrix_eqn_param eqn_param;
          libxsmm_matrix_arg arg_array[2];
          arg_array[0].primary = (void*)gout;
          arg_array[1].primary = (void*)in;
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = (void*)gin;

          kernel(&eqn_param);
        }
        void ref(T *gout, T *in, T *gin) {
#ifdef __AVX512F__
          int i;
          for(i = 0; i < ALIGNDOWN(N,16); i+=16) {
            auto vgout = _mm512_loadu_ps_auto(&gout[i]);
            auto vin_gelu = _mm512_loadu_ps_auto(&in[i]);
            auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
            //auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
            auto vout = _mm512_mul_ps(vgin_gelu, vgout);
            _mm512_storeu_ps_auto(&gin[i], vout);
          }
          if(i < N) {
            int rem = N - i;
            __mmask16 mask = (1 << rem) - 1;
            auto vgout = _mm512_maskz_loadu_ps_auto(mask, &gout[i]);
            auto vin_gelu = _mm512_maskz_loadu_ps_auto(mask, &in[i]);
            auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_TANH_PS_GELU_BWD(vin_gelu);
            //auto vgin_gelu = LIBXSMM_INTRINSICS_MM512_GELU_BWD_PS_MINIMAX3(vin_gelu);
            auto vout = _mm512_mul_ps(vgin_gelu, vgout);
            _mm512_mask_storeu_ps_auto(&gin[i], mask, vout);
          }
#else
          constexpr float PI = 3.14159265358979323846;
          for (int i = 0; i < N; i++) {
            float x = in[i];
            gin[i] = (float)gout[i] * (0.5 + 0.5 * erff(x/sqrtf(2.0)) + x/(sqrtf(2.0*PI))*expf(-0.5*x*x));
          }
#endif
        }
      protected:
        std::string hash_str() override {
          char hash[200];
          snprintf(hash, 200, "gelu_bwd_eqn_t%d_i%d", XsmmDtype<T>(), N);
          return std::string(hash);
        }
        void *build_kernel() override {
          auto dtype = XsmmDtype<T>();
          libxsmm_blasint ld = N;
          libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
          libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
          libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, N, 0, 0, dtype );
          libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_GELU_INV, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
          libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, N, 1, 0, dtype );
          libxsmm_matrix_eqn_tree_print( my_eqn0 );
          libxsmm_matrix_eqn_rpn_print( my_eqn0 );
          return (void*)libxsmm_dispatch_matrix_eqn( N, 1, &ld, dtype, my_eqn0 );
        }
      private:
        int N = 0;
        libxsmm_matrix_eqn_function kernel = NULL;
    };

  template<typename T>
    class DropOutFwdTPP {
      public:
        DropOutFwdTPP() { }
        DropOutFwdTPP(int N, float p) : N(N), p(p), kernel(1, N, N, N, XsmmDtype<T>(), XsmmDtype<T>(), LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BITMASK, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT) { }
        void operator()(T *in, void *rng_state, T *out, short *mask) {
          kernel((void*)in, rng_state, (void*)&p, (void*)out, (void*)mask);
        }
        void ref(T *in, void *rng_state, T *out, short *mask) {
          kernel((void*)in, rng_state, (void*)&p, (void*)out, (void*)mask);
        }
      private:
        int N = 0;
        float p;
        UnaryTPP kernel;
    };

  template<typename T>
    class DropOutBwdTPP {
      public:
        DropOutBwdTPP() { }
        DropOutBwdTPP(int N, float p) : N(N), p(p), kernel(1, N, N, N, XsmmDtype<T>(), XsmmDtype<T>(), LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_BITMASK, LIBXSMM_MELTW_TYPE_UNARY_DROPOUT_INV) { }
        void operator()(T *in, T *out, short *mask) {
          kernel((void*)in, (void*)mask, (void*)&p, (void*)out, (void*)NULL);
        }
        void ref(T *in, T *out, short *mask) {
          kernel((void*)in, (void*)mask, (void*)&p, (void*)out, (void*)NULL);
        }
      private:
        int N = 0;
        float p;
        UnaryTPP kernel;
    };

  template<typename Tin, typename Tout>
    class SoftMaxFwdTPP {
      public:
        SoftMaxFwdTPP() { }
        SoftMaxFwdTPP(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3), eqn1(S1, S2, S3) { }
        void operator()(Tin *in, Tout *out) {
          LIBXSMM_ALIGNED(float tmp[S1*S3], 64);
          for (int s2 = 0; s2 < S2; s2++) {
            eqn0(&in[s2*S3], tmp);
            eqn1(tmp, &out[s2*S3]);
          }
        }
        void ref(Tin *pinp, Tout *pout) {
          int s1, s2, s3;
          LIBXSMM_VLA_DECL(3, Tin, inp, pinp, S2, S3);
          LIBXSMM_VLA_DECL(3, Tout, out, pout, S2, S3);
#if defined(__AVX512F__)
          for (s2 = 0; s2 < S2; s2++) {
            float tmp[S1][S3];
            float max = upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
            float sum = 0.0;
            __m512 vmax = _mm512_set1_ps(max);
            __m512 vsum = _mm512_setzero_ps();

            for(s1 = 0; s1 < S1; s1++) {
              for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
                vmax = _mm512_max_ps(_mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax);
              }
              if (s3 < S3) {
                int rem = S3 - s3;
                __mmask16 mask = (1 << rem) - 1;
                vmax = _mm512_mask_max_ps(vmax, mask, _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax);
              }
            }
            max = _mm512_reduce_max_ps(vmax);
            vmax = _mm512_set1_ps(max);
            for (s1 = 0; s1 < S1; s1++) {
              for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
                __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(_mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax));
                _mm512_storeu_ps(&tmp[s1][s3], vz);
                vsum = _mm512_add_ps(vsum, vz);
              }
              if (s3 < S3) {
                int rem = S3 - s3;
                __mmask16 mask = (1 << rem) - 1;
                __m512 vz = LIBXSMM_INTRINSICS_MM512_EXP_PS_3DTS(_mm512_sub_ps(_mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3)), vmax));
                _mm512_mask_storeu_ps(&tmp[s1][s3], mask, vz);
                vsum = _mm512_mask_add_ps(vsum, mask, vsum, vz);
              }
            }
            sum = _mm512_reduce_add_ps(vsum);
            sum = 1.0 / sum;
            vsum = _mm512_set1_ps(sum);
            for (s1 = 0; s1 < S1; s1++) {
              for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
                _mm512_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), _mm512_mul_ps(vsum, _mm512_loadu_ps(&tmp[s1][s3])));
              }
              if (s3 < S3) {
                int rem = S3 - s3;
                __mmask16 mask = (1 << rem) - 1;
                _mm512_mask_storeu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), mask, _mm512_mul_ps(vsum, _mm512_maskz_loadu_ps(mask, &tmp[s1][s3])));
              }
            }
          }
#else
          for (s2 = 0; s2 < S2; s2++) {
            float tmp[S1][S3];
            float max = upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, 0, s2, 0, S2, S3));
            float sum = 0.0;
            for ( s1 = 0; s1 < S1; s1++) {
              for ( s3 = 0; s3 < S3; s3++) {
                float cur = upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
                if (max < cur) max = cur;
              }
            }
            for ( s1 = 0; s1 < S1; s1++) {
              for ( s3 = 0; s3 < S3; s3++) {
                float cur = upconvert_to_float(LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3));
                float z = expf(cur - max);
                tmp[s1][s3] = z;
                sum += z;
              }
            }
            sum = 1.0 / sum;
            for ( s1 = 0; s1 < S1; s1++) {
              for( s3 = 0; s3 < S3; s3++) {
                float cur = tmp[s1][s3] * sum;
                //libxsmm_rne_convert_fp32_bf16( &cur, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3), 1 );
                LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = cur;
              }
            }
          }
#endif
        }
        class Eqn0 : BaseTPP {
          public:
            Eqn0() { }
            Eqn0(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
              kernel = (libxsmm_matrix_eqn_function)get_kernel();
              initialized = true;
            }
            void operator()(Tin *in, float *out) {
              if (!initialized) return;
              libxsmm_matrix_eqn_param eqn_param;
              libxsmm_matrix_arg arg_array[1];
              arg_array[0].primary = (void*)in;
              eqn_param.inputs = arg_array;
              eqn_param.output.primary = (void*)out;

              kernel(&eqn_param);
            }
          protected:
            std::string hash_str() override {
              char hash[200];
              snprintf(hash, 200, "softmax_fwd_eqn0_ti%d_to%d_S1%d_S2%d_S3%d", XsmmDtype<Tin>(), LIBXSMM_DATATYPE_F32, S1, S2, S3);
              return std::string(hash);
            }
            void *build_kernel() override {
              auto dt_in = XsmmDtype<Tin>();
              libxsmm_blasint tmp_ld = S2;
              libxsmm_blasint ld = S2*S3;
              libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_EXP, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, ld, 0, 0, dt_in );
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_MAX, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, ld, 0, 0, dt_in );
              libxsmm_matrix_eqn_tree_print( my_eqn0 ); //printf
              return (void*)libxsmm_dispatch_matrix_eqn( S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn0 );
            }
          private:
            int S1, S2, S3;
            libxsmm_matrix_eqn_function kernel = NULL;
        };

        class Eqn1 : BaseTPP {
          public:
            Eqn1() { }
            Eqn1(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
              kernel = (libxsmm_matrix_eqn_function)get_kernel();
              initialized = true;
            }
            void operator()(float *in, Tout *out) {
              if (!initialized) return;
              libxsmm_matrix_eqn_param eqn_param;
              libxsmm_matrix_arg arg_array[1];
              arg_array[0].primary = (void*)in;
              eqn_param.inputs = arg_array;
              eqn_param.output.primary = (void*)out;

              kernel(&eqn_param);
            }
          protected:
            std::string hash_str() override {
              char hash[200];
              snprintf(hash, 200, "softmax_fwd_eqn1_ti%d_to%d_S1%d_S2%d_S3%d", LIBXSMM_DATATYPE_F32, XsmmDtype<Tout>(), S1, S2, S3);
              return std::string(hash);
            }
            void *build_kernel() override {
              auto dt_out = XsmmDtype<Tout>();
              libxsmm_blasint tmp_ld = S2;
              libxsmm_blasint ld = S2*S3;
              libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_RECIPROCAL, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn1, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
              /*libxsmm_matrix_eqn_tree_print( my_eqn1 );*/
              return (void*)libxsmm_dispatch_matrix_eqn( S3, S1, &ld, dt_out, my_eqn1 );
            }
          private:
            int S1, S2, S3;
            libxsmm_matrix_eqn_function kernel = NULL;
        };
      private:
        int S1, S2, S3;
        Eqn0 eqn0;
        Eqn1 eqn1;
    };

  template<typename T1, typename T2, typename T3>
    class SoftMaxBwdTPP {
      public:
        SoftMaxBwdTPP() { }
        SoftMaxBwdTPP(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3), eqn0(S1, S2, S3, 0), eqn1(S1, S2, S3, 1) { }
        void operator()(T1 *gin, T2 *gout, T3 *out) {
          LIBXSMM_ALIGNED(float tmp[S1*S3], 64);
          for (int s2 = 0; s2 < S2; s2++) {
            libxsmm_matrix_eqn_param eqn_param;
            libxsmm_matrix_arg arg_array[2];
            arg_array[0].primary = (void*)&gout[s2*S3];
            arg_array[1].primary = (void*)&out[s2*S3];
            eqn_param.inputs = arg_array;
            eqn_param.output.primary = (void*)tmp;
            eqn0(&eqn_param);

            arg_array[0].primary = (void*)tmp;
            eqn_param.output.primary = (void*)&gin[s2*S3];
            eqn1(&eqn_param);
          }
        }
        void ref(T1 *pgradinp, T2 *pgradout, T3 *pout) {
          int s1, s2, s3;
          LIBXSMM_VLA_DECL(3, T1, ginp, pgradinp, S2, S3);
          LIBXSMM_VLA_DECL(3, T2, gout, pgradout, S2, S3);
          LIBXSMM_VLA_DECL(3, T3, out, pout, S2, S3);
#if defined(__AVX512F__)
          for (s2 = 0; s2 < S2; s2++) {
            float sum = 0.0;
            __m512 vsum = _mm512_setzero_ps();
            for (s1 = 0; s1 < S1; s1++) {
              for (s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
                __m512 vgo = _mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
                __m512 vo = _mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
                vsum = _mm512_fmadd_ps(vgo, vo, vsum);
              }
              if (s3 < S3) {
                int rem = S3 - s3;
                __mmask16 mask = (1 << rem) - 1;
                __m512 vgo = _mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3));
                __m512 vo = _mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
                vsum = _mm512_fmadd_ps(vgo, vo, vsum);
              }
            }
            sum = _mm512_reduce_add_ps(vsum);
            vsum = _mm512_set1_ps(sum);
            for (s1 = 0; s1 < S1; s1++) {
              for(s3 = 0; s3 < ALIGNDOWN(S3, 16); s3+=16) {
                __m512 tmp = _mm512_sub_ps(_mm512_loadu_ps(&LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)), vsum);
                _mm512_storeu_ps(&LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3), _mm512_mul_ps(_mm512_loadu_ps_auto(&LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)), tmp));
              }
              if (s3 < S3) {
                int rem = S3 - s3;
                __mmask16 mask = (1 << rem) - 1;
                __m512 tmp = _mm512_sub_ps(_mm512_maskz_loadu_ps(mask, &LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3)), vsum);
                _mm512_mask_storeu_ps(&LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3), mask, _mm512_mul_ps(_mm512_maskz_loadu_ps_auto(mask, &LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)), tmp));
              }
            }
          }
#else
          for (s2 = 0; s2 < S2; s2++) {
            float sum = 0.0;
            for (s1 = 0; s1 < S1; s1++) {
              for (s3 = 0; s3 < S3; s3++) {
                sum += LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) * upconvert_to_float(LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3));
              }
            }
            for (s1 = 0; s1 < S1; s1++) {
              for (s3 = 0; s3 < S3; s3++) {
                LIBXSMM_VLA_ACCESS(3, ginp, s1, s2, s3, S2, S3) = upconvert_to_float(LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3)) * (LIBXSMM_VLA_ACCESS(3, gout, s1, s2, s3, S2, S3) - sum);
              }
            }
          }
#endif
        }

        class Eqn : BaseTPP {
          public:
            Eqn() { }
            Eqn(int S1, int S2, int S3, int eqn_no) : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
              kernel = (libxsmm_matrix_eqn_function)get_kernel();
              initialized = true;
            }
            void operator()(libxsmm_matrix_eqn_param *eqn_param) {
              if (!initialized) return;
              kernel(eqn_param);
            }
          protected:
            std::string hash_str() override {
              char hash[200];
              snprintf(hash, 200, "softmax_bwd_eqn%d_t1%d_t2%d_t3%d_S1%d_S2%d_S3%d", eqn_no, XsmmDtype<T2>(), XsmmDtype<T3>(),LIBXSMM_DATATYPE_F32, S1, S2, S3);
              return std::string(hash);
            }
            void *build_kernel() override {
              auto dt_1 = XsmmDtype<T1>();
              auto dt_2 = XsmmDtype<T2>();
              auto dt_3 = XsmmDtype<T3>();
              libxsmm_blasint tmp_ld = S3;
              libxsmm_blasint ld = S2*S3;
              libxsmm_matrix_eqn_function func;
              if (eqn_no == 0) {
                libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 0, 0, dt_2 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 1, 0, dt_3 );
                libxsmm_matrix_eqn_tree_print( my_eqn2 ); //printf
                func = libxsmm_dispatch_matrix_eqn( S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2 );
              } else if (eqn_no == 1) {
                libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
#if 1
                libxsmm_matrix_eqn_push_back_ternary_op( my_eqn3, LIBXSMM_MELTW_TYPE_TERNARY_NMULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_0 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, ld, 1, 0, dt_3 );
#else
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, ld, 1, 0, dt_3 );
                libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_unary_op( my_eqn3, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 0, 0, LIBXSMM_DATATYPE_F32 );
#endif
                libxsmm_matrix_eqn_tree_print( my_eqn3 );
                func = libxsmm_dispatch_matrix_eqn( S3, S1, &ld, dt_1, my_eqn3 );
              } else {
                PCL_ASSERT(false, "Should not come here\n");
              }
              return (void*)func;
            }
          private:
            int S1, S2, S3, eqn_no;
            libxsmm_matrix_eqn_function kernel = NULL;
        };

      private:
        int S1, S2, S3;
        Eqn eqn0, eqn1;
    };

  template<typename T>
    class LayerNormFwdTPP {
      public:
        LayerNormFwdTPP() { }
        LayerNormFwdTPP(int S1, int S2, int S3, float eps) : S1(S1), S2(S2), S3(S3), eps(eps),
        reduce_cols_kernel(S1, S3, S2*S3, S3, XsmmDtype<T>(), LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_X2_OP_ADD),
        reduce_rows_kernel(1, S3, S3, 1, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_ROWS, LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD),
        eqn(S1, S2, S3) { }
        void operator()(T *inp, T *gamma, T *beta, float *mean, float *var, T *out) {
          LIBXSMM_ALIGNED(float tmp[2*S3], 64);
          const float c = 1.0/((float)S1*S3);
          float m, v, s, b;
          libxsmm_matrix_eqn_param eqn_param;
          libxsmm_matrix_arg  arg_array[5];
          eqn_param.inputs = arg_array;
          arg_array[1].primary = &s;
          arg_array[2].primary = &b;
          arg_array[3].primary = (void*)gamma;
          arg_array[4].primary = (void*)beta;
          for (int s2 = 0; s2 < S2; s2++) {
            reduce_cols_kernel((void*)&inp[s2*S3], (void*)tmp);
            reduce_rows_kernel((void*)tmp, (void*)&m);
            reduce_rows_kernel((void*)&tmp[S3], (void*)&v);
            m = m * c;
            v = v * c;
            v = LIBXSMM_MAX(v - m * m, 0.0f);
            v = 1.0f / ((float)sqrt(v+eps));
            mean[s2] = m;
            var[s2] = v;
            s = v;
            b = -1.0 * v * m;
            arg_array[0].primary = (void*)&inp[s2*S3];
            eqn_param.output.primary = (void*)&out[s2*S3];
            eqn(&eqn_param);
          }
        }
        void ref(T *pinp, T *pgamma, T *pbeta, float *mean, float *var, T *pout) {
          int s1, s2, s3;
          LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
          LIBXSMM_VLA_DECL(3, T, out, pout, S2, S3);
          LIBXSMM_VLA_DECL(2, T, gamma, pgamma, S3);
          LIBXSMM_VLA_DECL(2, T, beta, pbeta, S3);
          for (s2 = 0; s2 < S2; s2++) {
            float m = 0;
            float v = 0;
            float c = 1.0 / (S1*S3);
            for (s1 = 0; s1 < S1; s1++) {
              for( s3 = 0; s3 < S3; s3++) {
                m += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
                v += LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
              }
            }
            m = m * c;
            v = v * c;
            v = LIBXSMM_MAX(v - m * m, 0.0f);
            v = 1.0f / ((float)sqrt(v+eps));
            mean[s2] = m;
            var[s2] = v;
            float s = v;
            float b = -1.0 * v * m;
            for (s1 = 0; s1 < S1; s1++) {
              for (s3 = 0; s3 < S3; s3++) {
                LIBXSMM_VLA_ACCESS(3, out, s1, s2, s3, S2, S3) = (LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) * s + b) * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) + LIBXSMM_VLA_ACCESS(2, beta, s1, s3, S3);
              }
            }
          }
        }
        class Eqn : BaseTPP {
          public:
            Eqn() { }
            Eqn(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3) {
              kernel = (libxsmm_matrix_eqn_function)get_kernel();
              initialized = true;
            }
            void operator()(libxsmm_matrix_eqn_param *eqn_param) {
              if (!initialized) return;
              kernel(eqn_param);
            }
          protected:
            std::string hash_str() override {
              char hash[200];
              snprintf(hash, 200, "layernorm_fwd_eqn_t%d_S1%d_S2%d_S3%d", XsmmDtype<T>(), S1, S2, S3);
              return std::string(hash);
            }
            void *build_kernel() override {
              auto in_dt = XsmmDtype<T>();
              auto out_dt = XsmmDtype<T>();
              libxsmm_blasint tmp_ld = 1;
              libxsmm_blasint tmp_ld2 = S3;
              libxsmm_blasint ld = S2*S3;
              libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
              libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, ld, 0, 0, in_dt );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, tmp_ld, 1, 0, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, tmp_ld, 2, 0, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, tmp_ld2, 3, 0, in_dt );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, S3, S1, tmp_ld2, 4, 0, in_dt );
              libxsmm_matrix_eqn_tree_print( my_eqn0 ); //printf
              return (void*)libxsmm_dispatch_matrix_eqn( S3, S1, &ld, out_dt, my_eqn0 );
            }
          private:
            int S1, S2, S3;
            libxsmm_matrix_eqn_function kernel = NULL;
        };

      private:
        int S1, S2, S3;
        float eps;
        UnaryTPP reduce_cols_kernel;
        UnaryTPP reduce_rows_kernel;
        Eqn eqn;
    };

  template<typename T>
    class LayerNormBwdTPP {
      public:
        LayerNormBwdTPP() { }
        LayerNormBwdTPP(int S1, int S2, int S3) : S1(S1), S2(S2), S3(S3),
        dgamma_func(S1, S2, S3, 1), dbeta_func(S1, S2, S3, 2), db_func(S1, S2, S3, 3), ds_func(S1, S2, S3, 4), din_func(S1, S2, S3, 5) { }
        void operator()(T *dout, T *inp, float *mean, float *var, T *gamma, T *din, float *dgamma, float *dbeta) {
          float a, b, c, db, ds;
          const float scale = 1.0f / ((float)S1*S3);
          libxsmm_matrix_eqn_param eqn_param;
          libxsmm_matrix_arg arg_array[8];
          eqn_param.inputs = arg_array;

          arg_array[1].primary = &a;
          arg_array[2].primary = &b;
          arg_array[4].primary = (void*)dgamma;
          arg_array[5].primary = (void*)dbeta;
          arg_array[6].primary = (void*)gamma;
          arg_array[7].primary = &c;

          for (int s2 = 0; s2 < S2; s2++) {
            a = var[s2];
            b = -a*mean[s2];
            arg_array[0].primary = (void*)&inp[s2*S3];
            arg_array[3].primary = (void*)&dout[s2*S3];

            eqn_param.output.primary = &ds;
            ds_func(&eqn_param);

            eqn_param.output.primary = &db;
            db_func(&eqn_param);

            eqn_param.output.primary = (void*)dgamma;
            dgamma_func(&eqn_param);

            eqn_param.output.primary = (void*)dbeta;
            dbeta_func(&eqn_param);

            b = (db * mean[s2] - ds) * a * a * a * scale;
            c = -b * mean[s2] - db * a * scale;

            eqn_param.output.primary = (void*)&din[s2*S3];
            din_func(&eqn_param);
          }
        }
        void ref(T *pdout, T *pinp, float *mean, float *var, T *pgamma, T *pdin, float *pdgamma, float *pdbeta) {
          int s1, s2, s3;
          LIBXSMM_VLA_DECL(3, T, din, pdin, S2, S3);
          LIBXSMM_VLA_DECL(3, T, inp, pinp, S2, S3);
          LIBXSMM_VLA_DECL(3, T, dout, pdout, S2, S3);
          LIBXSMM_VLA_DECL(2, T, gamma, pgamma, S3);
          LIBXSMM_VLA_DECL(2, float, dgamma, pdgamma, S3);
          LIBXSMM_VLA_DECL(2, float, dbeta, pdbeta, S3);
          for (s2 = 0; s2 < S2; s2++) {
            float a = var[s2], c;
            float b = -a*mean[s2];
            float ds = 0.0f;
            float db = 0.0f;
            float scale = 1.0f / (S1 * S3);
            for (s1 = 0; s1 < S1; s1++) {
              for (s3 = 0; s3 < S3; s3++) {
                LIBXSMM_VLA_ACCESS(2, dgamma, s1, s3, S3) += (a * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + b) * LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
                LIBXSMM_VLA_ACCESS(2, dbeta, s1, s3, S3) += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3);
                ds += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3);
                db += LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3) * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3);
              }
            }
            b = (db * mean[s2] - ds) * a * a * a * scale;
            c = -b * mean[s2] - db * a * scale;
            for (s1 = 0; s1 < S1; s1++) {
              for (s3 = 0; s3 < S3; s3++) {
                LIBXSMM_VLA_ACCESS(3, din, s1, s2, s3, S2, S3) = LIBXSMM_VLA_ACCESS(3, dout, s1, s2, s3, S2, S3)  * a * LIBXSMM_VLA_ACCESS(2, gamma, s1, s3, S3) + b * LIBXSMM_VLA_ACCESS(3, inp, s1, s2, s3, S2, S3) + c;
              }
            }
          }
        }

        class Eqn : BaseTPP {
          public:
            Eqn() { }
            Eqn(int S1, int S2, int S3, int eqn_no) : S1(S1), S2(S2), S3(S3), eqn_no(eqn_no) {
              kernel = (libxsmm_matrix_eqn_function)get_kernel();
              initialized = true;
            }
            void operator()(libxsmm_matrix_eqn_param *eqn_param) {
              if (!initialized) return;
              kernel(eqn_param);
            }
          protected:
            std::string hash_str() override {
              char hash[200];
              snprintf(hash, 200, "layernorm_bwd_eqn%d_t%d_S1%d_S2%d_S3%d", eqn_no, XsmmDtype<T>(), S1, S2, S3);
              return std::string(hash);
            }
            void *build_kernel() override {
              auto in_dt = XsmmDtype<T>();
              //auto out_dt = XsmmDtype<T>();
              libxsmm_blasint tmp_ld = S3;
              libxsmm_blasint tmp_ld2 = 1;
              libxsmm_blasint ld = S2*S3;
              libxsmm_matrix_eqn_function func = NULL;
              if (eqn_no == 1) {
                /* dgamma function  */
                libxsmm_blasint my_eqn1 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_ternary_op( my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_ternary_op( my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, S3, S1, ld, 0, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, S3, S1, ld, 3, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, S3, S1, tmp_ld, 4, 0, LIBXSMM_DATATYPE_F32 );
                /*libxsmm_matrix_eqn_tree_print( my_eqn1 );*/
                func = libxsmm_dispatch_matrix_eqn( S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn1 );
              } else if (eqn_no == 2) {
                /* dbeta function  */
                libxsmm_blasint my_eqn2 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, ld, 3, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, S3, S1, tmp_ld, 5, 0, LIBXSMM_DATATYPE_F32 );
                /*libxsmm_matrix_eqn_tree_print( my_eqn1 );*/
                func = libxsmm_dispatch_matrix_eqn( S3, S1, &tmp_ld, LIBXSMM_DATATYPE_F32, my_eqn2 );
              } else if (eqn_no == 3) {
                /* db equation */
                libxsmm_blasint my_eqn3 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn3, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, ld, 3, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn3, S3, S1, tmp_ld, 6, 0, in_dt );
                func = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn3 );
              } else if (eqn_no == 4) {
                /* ds equation */
                libxsmm_blasint my_eqn4 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL_AND_REDUCE_TO_SCALAR_OP_ADD, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn4, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn4, S3, S1, ld, 3, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn4, S3, S1, tmp_ld, 6, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn4, S3, S1, ld, 0, 0, in_dt );
                func = libxsmm_dispatch_matrix_eqn( 1, 1, &tmp_ld2, LIBXSMM_DATATYPE_F32, my_eqn4 );
              } else if (eqn_no == 5) {
                /* din equation */
                libxsmm_blasint my_eqn5 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_ternary_op( my_eqn5, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn5, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn5, S3, S1, tmp_ld, 6, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn5, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn5, S3, S1, ld, 3, 0, in_dt );
                libxsmm_matrix_eqn_push_back_ternary_op( my_eqn5, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_2 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn5, S3, S1, ld, 0, 0, in_dt );
                libxsmm_matrix_eqn_push_back_arg( my_eqn5, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn5, 1, 1, 1, 7, 0, LIBXSMM_DATATYPE_F32 );
                func = libxsmm_dispatch_matrix_eqn( S3, S1, &ld, in_dt, my_eqn5 );
              } else {
                PCL_ASSERT(false, "LayerNormBwdTPP: invalid eqn. number %d\n", eqn_no);
              }
              return (void*)func;
            }
          private:
            int S1, S2, S3, eqn_no;
            libxsmm_matrix_eqn_function kernel = NULL;
        };
      private:
        int S1, S2, S3;
        Eqn dgamma_func, dbeta_func, db_func, ds_func, din_func;
    };

  class SplitSGDTPP : public BaseTPP {
    public:
      SplitSGDTPP() { }
      SplitSGDTPP(int N) : N(N){
        kernel = (libxsmm_matrix_eqn_function)get_kernel();
        initialized = true;
      }
      void operator()(bfloat16 *hi, bfloat16 *lo, bfloat16 *grad, float lr) {
        if (!initialized) return;
        libxsmm_matrix_eqn_param eqn_param;
        libxsmm_matrix_arg arg_array[4];
        arg_array[0].primary = (void*)lo;
        arg_array[1].primary = (void*)hi;
        arg_array[2].primary = (void*)&lr;
        arg_array[3].primary = (void*)grad;
        eqn_param.inputs = arg_array;
        eqn_param.output.primary = (void*)lo;
        auto offset = (long long) ((char*)hi - (char*)lo);
        eqn_param.output.secondary = (void*)offset;

        kernel(&eqn_param);
      }
      void ref(bfloat16 *hi, bfloat16 *lo, bfloat16 *grad, float lr) {
        auto dwt = (libxsmm_bfloat16 *)grad;
        auto out_hi = (libxsmm_bfloat16 *)hi;
        auto out_lo = (libxsmm_bfloat16 *)lo;
        for (int i = 0; i < N; i++) {
          union libxsmm_bfloat16_hp bf16_hp;
          union libxsmm_bfloat16_hp bf16_wt;
          bf16_wt.i[0] = 0;
          bf16_wt.i[1] = dwt[i];
          bf16_hp.i[0] = out_lo[i];
          bf16_hp.i[1] = out_hi[i];
          bf16_hp.f = bf16_wt.f * lr + bf16_hp.f;
          out_lo[i] = bf16_hp.i[0];
          out_hi[i] = bf16_hp.i[1];
        }
      }
    protected:
      std::string hash_str() override {
        char hash[200];
        snprintf(hash, 200, "split_sgd_eqn_i%d", N);
        return std::string(hash);
      }
      void *build_kernel() override {
        libxsmm_blasint ld = N;
        libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
        libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
        libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
        libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, ld, 3, 0, LIBXSMM_DATATYPE_BF16 ); /* This is the "gradient" weights   */
        libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, 1, 2, 0, LIBXSMM_DATATYPE_F32 );   /* This is the scalar learning rate */
        libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_PACK, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
        libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, ld, 0, 0, LIBXSMM_DATATYPE_I16 );  /* This is the tensor with lo bits  */
        libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, ld, 1, 0, LIBXSMM_DATATYPE_I16 );  /* This is the tensor with hi bits  */
        libxsmm_matrix_eqn_tree_print( my_eqn0 );
        libxsmm_matrix_eqn_rpn_print( my_eqn0 );
        auto func0 = libxsmm_dispatch_matrix_eqn( N, 1, &ld, LIBXSMM_DATATYPE_I16, my_eqn0 );
        return (void*)func0;
      }
    private:
      int N = 0;
      libxsmm_matrix_eqn_function kernel = NULL;
  };

  template<typename Tin, typename Tout, typename Tind>
    class EmbBagFwdTPP : public BaseTPP {
      public:
        EmbBagFwdTPP() { }
        EmbBagFwdTPP(int E) : E(E) {
          kernel = (libxsmm_meltwfunction_reduce_cols_idx)get_kernel();
          initialized = true;
        }
        void operator()(Tout *output, Tin *weight, Tind *input, int N) {
          if (!initialized) return;
          libxsmm_meltw_reduce_cols_idx_param params;
          params.n = N;
          params.ind_ptr = (void*)input;
          params.inp_ptr = (void*)weight;
          params.out_ptr = (void*)output;
          kernel( &params );
        }
        void ref(Tout *output, Tin *weight, Tind *input, int N) {
          for (long v = 0; v < E; v++)
            output[v] = 0;
          for (long s = 0; s < N; s++)
          {
            auto ind = input[s];
            for (long v = 0; v < E; v++)
              output[v] += weight[ind*E+v];
          }
        }
      protected:
        std::string hash_str() override {
          char hash[200];
          snprintf(hash, 200, "emb_bag_fwd_eqn_t%d_t%d_t%d_i%d", XsmmDtype<Tin>(), XsmmDtype<Tout>(), XsmmDtype<Tind>(), E);
          return std::string(hash);
        }
        void *build_kernel() override {
          auto dt_in = XsmmDtype<Tin>();
          auto dt_out = XsmmDtype<Tout>();
          auto dt_ind = XsmmDtype<Tind>();
          libxsmm_blasint ld = E;
          setenv("LOAD_ACCS_REDUCE_COLS_IDX", "0", 1);
          return (void*)libxsmm_dispatch_meltw_reduce_cols_idx(E, &ld, &ld, dt_in, dt_out, dt_ind);
        }
      private:
        int E;
        libxsmm_meltwfunction_reduce_cols_idx kernel = NULL;
    };

  template<typename Tin, typename Tout>
    class EmbBagBwdTPP {
      public:
        EmbBagBwdTPP() { }
        EmbBagBwdTPP(int E) : E(E), kernel(0, E, E, E, XsmmDtype<Tin>(), XsmmDtype<Tout>(), XsmmDtype<Tout>(), LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_MELTW_TYPE_UNARY_REPLICATE_COL_VAR) { }
        void operator()(Tin *in, Tout *out, uint64_t N) {
          kernel((void*)in, (void*)out, (void*)&N);
        }
        void ref(Tin *in, Tout *out, uint64_t N) {
          for (uint64_t i = 0; i < N; i++) {
            for (int v = 0; v < E; v++) {
              out[i*E+v] = in[v];
            }
          }
        }
      private:
        int E;
        UnaryTPP kernel;
    };

  template<typename T>
    class FusedAdamWTPP {
      public:
        FusedAdamWTPP() { }
        FusedAdamWTPP(int N, float beta1, float beta2, float weight_decay, float eps) : N(N), beta1(beta1), beta2(beta2), weight_decay(weight_decay), eps(eps), eqn0(this, 0), eqn1(this, 1), eqn2(this, 2) { }
        void operator()(T *data, T *grad, T *exp_avg, T *exp_avg_sq, float step_size, float lr) {
          float beta1_1 = 1.0f - beta1;
          float beta2_1 = 1.0f - beta2;
          float lrwd_1 = 1.0f - lr * weight_decay;
          libxsmm_matrix_eqn_param eqn_param;
          libxsmm_matrix_arg arg_array[6];
          arg_array[0].primary = (void*)grad;
          arg_array[1].primary = (void*)&beta1_1;
          arg_array[2].primary = (void*)exp_avg;
          arg_array[3].primary = (void*)&beta1;
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = (void*)exp_avg;
          eqn0(&eqn_param);

          //arg_array[0].primary = (void*)grad;
          arg_array[1].primary = (void*)&beta2_1;
          arg_array[2].primary = (void*)exp_avg_sq;
          arg_array[3].primary = (void*)&beta2;
          eqn_param.output.primary = (void*)exp_avg_sq;
          eqn1(&eqn_param);

          arg_array[0].primary = (void*)exp_avg_sq;
          arg_array[1].primary = (void*)&eps;
          arg_array[2].primary = (void*)exp_avg;
          arg_array[3].primary = (void*)&step_size;
          arg_array[4].primary = (void*)data;
          arg_array[5].primary = (void*)&lrwd_1;
          eqn_param.output.primary = (void*)data;
          eqn2(&eqn_param);
        }

        void ref(T *data, T *grad, T *exp_avg, T *exp_avg_sq, float step_size, float lr) {
          long sz = N;
          float beta1_1 = 1.0f - beta1;
          float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
          for (long i = 0; i < sz; i++) {
            auto avg_i = exp_avg[i];
            auto avg_sq_i = exp_avg_sq[i];
            auto grad_i = grad[i];
            auto data_i = data[i];
            avg_i = avg_i * beta1 + grad_i * beta1_1;
            avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
            auto denom = sqrtf(avg_sq_i) + eps;
            data_i = data_i - step_size * (avg_i / denom);
            if (weight_decay > 0.0)
              data_i = data_i - data_i * lr * weight_decay;
            exp_avg[i] = avg_i;
            exp_avg_sq[i] = avg_sq_i;
            data[i] = data_i;
          }
#else
          auto vbeta1 = _mm512_set1_ps(beta1);
          auto vbeta1_1 = _mm512_set1_ps(beta1_1);
          auto vbeta2 = _mm512_set1_ps(beta2);
          auto vbeta2_1 = _mm512_set1_ps(beta2_1);
          auto veps = _mm512_set1_ps(eps);
          auto vstep_size = _mm512_set1_ps(step_size);
          auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
          long i;
          for (i = 0; i < ALIGNDOWN(sz, 16); i+=16) {
            auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
            auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
            auto grad_i = _mm512_loadu_ps(&grad[i]);
            auto data_i = _mm512_loadu_ps(&data[i]);
            avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
            avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
            auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
            data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
            //if (weight_decay > 0.0)
            data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
            _mm512_storeu_ps(&exp_avg[i], avg_i);
            _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
            _mm512_storeu_ps(&data[i], data_i);
          }
          if( i < sz) {
            int rem = sz - i;
            __mmask16 mask = (1 << rem) - 1;
            auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
            auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
            auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
            auto data_i = _mm512_maskz_loadu_ps(mask, &data[i]);
            avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
            avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
            auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
            data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
            //if (weight_decay > 0.0)
            data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
            _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
            _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
            _mm512_mask_storeu_ps(&data[i], mask, data_i);
          }
#endif

        }
        class Eqn : BaseTPP {
          public:
            Eqn() { }
            Eqn(FusedAdamWTPP *p, int eqn_no) : p(p), eqn_no(eqn_no) {
              kernel = (libxsmm_matrix_eqn_function)get_kernel();
              initialized = true;
            }
            void operator()(libxsmm_matrix_eqn_param *eqn_param) {
              if (!initialized) return;
              kernel(eqn_param);
            }
          protected:
            std::string hash_str() override {
              char hash[200];
              snprintf(hash, 200, "fused_adamw_eqn%d_t%d_n%d_wd%d", eqn_no, XsmmDtype<T>(), p->N, (p->weight_decay == 0.0 ? 0 : 1));
              return std::string(hash);
            }
            void *build_kernel() override {
              auto in_dt = XsmmDtype<T>();
              libxsmm_blasint ld = p->N;
              auto N = p->N;
              int use_wd = p->weight_decay == 0.0 ? 0 : 1;
              libxsmm_matrix_eqn_function func;
              if (eqn_no == 0) {
                // Equation for exp_avg
                auto my_eqn0 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, ld, 2, 0, in_dt ); // avg_i
                libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32 ); // beta1
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, ld, 0, 0, in_dt ); // grad_i
                libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 ); // beta1_1
                libxsmm_matrix_eqn_tree_print( my_eqn0 );
                libxsmm_matrix_eqn_rpn_print( my_eqn0 );
                func = libxsmm_dispatch_matrix_eqn(N, 1, &ld, in_dt, my_eqn0 );
              } else if (eqn_no == 1) {
                // Equation for exp_avg_sq
                auto my_eqn1 = libxsmm_matrix_eqn_create();
                libxsmm_matrix_eqn_push_back_ternary_op( my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, N, 1, ld, 2, 0, in_dt ); // avg_sq_i
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32 ); // beta2
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, N, 1, ld, 0, 0, in_dt ); // grad_i
                libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 ); // beta2_1
                libxsmm_matrix_eqn_tree_print( my_eqn1 );
                libxsmm_matrix_eqn_rpn_print( my_eqn1 );
                func = libxsmm_dispatch_matrix_eqn( N, 1, &ld, in_dt, my_eqn1 );
              } else if (eqn_no == 2) {
                // Equation for data_i (with decay)
                auto my_eqn2 = libxsmm_matrix_eqn_create();
                if (use_wd == 1) {
                  libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
                }
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, N, 1, ld, 4, 0, in_dt ); //data_i
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, N, 1, ld, 2, 0, in_dt ); // avg_i
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_unary_op( my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, N, 1, ld, 0, 0, in_dt ); // avg_sq_i
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 ); // eps
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32 ); // step_size
                if (use_wd == 1) {
                  libxsmm_matrix_eqn_push_back_arg( my_eqn2, 1, 1, 1, 5, 0, LIBXSMM_DATATYPE_F32 ); // this scalar is (1-lr*weight_decay)
                }
                libxsmm_matrix_eqn_tree_print( my_eqn2 );
                libxsmm_matrix_eqn_rpn_print( my_eqn2 );
                func = libxsmm_dispatch_matrix_eqn( N, 1, &ld, in_dt, my_eqn2 );
              } else {
                PCL_ASSERT(false, "Should not come here\n");
              }
              return (void*)func;
            }
          private:
            FusedAdamWTPP *p;
            int eqn_no;
            libxsmm_matrix_eqn_function kernel = NULL;
        };

      private:
        int N = 0;
        float beta1, beta2, weight_decay, eps;
        Eqn eqn0, eqn1, eqn2;
        friend class Eqn;
    };

  class FusedSplitAdamWTPP {
    public:
      typedef bfloat16 T;
      FusedSplitAdamWTPP() { }
      FusedSplitAdamWTPP(int N, float beta1, float beta2, float weight_decay, float eps) : N(N), beta1(beta1), beta2(beta2), weight_decay(weight_decay), eps(eps), eqn0(this, 0), eqn1(this, 1), eqn2(this, 2) { }
      void operator()(T *hi, T *lo, T *grad, T *exp_avg, T *exp_avg_sq, float step_size, float lr) {
        float beta1_1 = 1.0f - beta1;
        float beta2_1 = 1.0f - beta2;
        float lrwd_1 = 1.0f - lr * weight_decay;
        libxsmm_matrix_eqn_param eqn_param;
        libxsmm_matrix_arg arg_array[7];
        arg_array[0].primary = (void*)grad;
        arg_array[1].primary = (void*)&beta1_1;
        arg_array[2].primary = (void*)exp_avg;
        arg_array[3].primary = (void*)&beta1;
        eqn_param.inputs = arg_array;
        eqn_param.output.primary = (void*)exp_avg;
        eqn0(&eqn_param);

        //arg_array[0].primary = (void*)grad;
        arg_array[1].primary = (void*)&beta2_1;
        arg_array[2].primary = (void*)exp_avg_sq;
        arg_array[3].primary = (void*)&beta2;
        eqn_param.output.primary = (void*)exp_avg_sq;
        eqn1(&eqn_param);

        arg_array[0].primary = (void*)exp_avg_sq;
        arg_array[1].primary = (void*)&eps;
        arg_array[2].primary = (void*)exp_avg;
        arg_array[3].primary = (void*)&step_size;
        arg_array[4].primary = (void*)lo;
        arg_array[5].primary = (void*)hi;
        arg_array[6].primary = (void*)&lrwd_1;
        eqn_param.output.primary = (void*)lo;
        auto offset = (long long) ((char*)hi - (char*)lo);
        eqn_param.output.secondary = (void*)offset;
        eqn2(&eqn_param);
      }

      void ref(T *hi, T *lo, T *grad, T *exp_avg, T *exp_avg_sq, float step_size, float lr) {
        long sz = N;
        float beta1_1 = 1.0f - beta1;
        float beta2_1 = 1.0f - beta2;
#ifndef __AVX512F__
        for (long i = 0; i < sz; i++) {
          union libxsmm_bfloat16_hp data_hp;
          float avg_i = exp_avg[i];
          float avg_sq_i = exp_avg_sq[i];
          float grad_i = grad[i];
          data_hp.i[0] = lo[i];
          data_hp.i[1] = hi[i];
          float data_i = data_hp.f;

          avg_i = avg_i * beta1 + grad_i * beta1_1;
          avg_sq_i = avg_sq_i * beta2 + grad_i * grad_i * beta2_1;
          auto denom = sqrtf(avg_sq_i) + eps;
          data_i = data_i - step_size * (avg_i / denom);
          if (weight_decay > 0.0)
            data_i = data_i - data_i * lr * weight_decay;
          exp_avg[i] = avg_i;
          exp_avg_sq[i] = avg_sq_i;
          data_hp.f = data_i;
          lo[i] = data_hp.i[0];
          hi[i] = data_hp.i[1];
        }
#else
        auto vbeta1 = _mm512_set1_ps(beta1);
        auto vbeta1_1 = _mm512_set1_ps(beta1_1);
        auto vbeta2 = _mm512_set1_ps(beta2);
        auto vbeta2_1 = _mm512_set1_ps(beta2_1);
        auto veps = _mm512_set1_ps(eps);
        auto vstep_size = _mm512_set1_ps(step_size);
        auto vweight_decay = _mm512_set1_ps(lr * weight_decay);
        long i;
        for (i = 0; i < ALIGNDOWN(sz, 16); i+=16) {
          auto avg_i = _mm512_loadu_ps(&exp_avg[i]);
          auto avg_sq_i = _mm512_loadu_ps(&exp_avg_sq[i]);
          auto grad_i = _mm512_loadu_ps(&grad[i]);
          auto data_i = _mm512_split_loadu_ps(&hi[i], &lo[i]);
          avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
          avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
          auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
          data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
          if (weight_decay > 0.0)
            data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
          _mm512_storeu_ps(&exp_avg[i], avg_i);
          _mm512_storeu_ps(&exp_avg_sq[i], avg_sq_i);
          _mm512_split_storeu_ps(&hi[i], &lo[i], data_i);
        }
        if( i < sz) {
          int rem = sz - i;
          __mmask16 mask = (1 << rem) - 1;
          auto avg_i = _mm512_maskz_loadu_ps(mask, &exp_avg[i]);
          auto avg_sq_i = _mm512_maskz_loadu_ps(mask, &exp_avg_sq[i]);
          auto grad_i = _mm512_maskz_loadu_ps(mask, &grad[i]);
          auto data_i = _mm512_maskz_split_loadu_ps(mask, &hi[i], &lo[i]);
          avg_i = _mm512_add_ps(_mm512_mul_ps(avg_i, vbeta1), _mm512_mul_ps(grad_i, vbeta1_1));
          avg_sq_i = _mm512_add_ps(_mm512_mul_ps(avg_sq_i, vbeta2), _mm512_mul_ps(_mm512_mul_ps(grad_i, grad_i), vbeta2_1));
          auto denom = _mm512_add_ps(_mm512_sqrt_ps(avg_sq_i), veps);
          data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(vstep_size, _mm512_div_ps(avg_i, denom)));
          if (weight_decay > 0.0)
            data_i = _mm512_sub_ps(data_i, _mm512_mul_ps(data_i, vweight_decay));
          _mm512_mask_storeu_ps(&exp_avg[i], mask, avg_i);
          _mm512_mask_storeu_ps(&exp_avg_sq[i], mask, avg_sq_i);
          _mm512_mask_split_storeu_ps(&hi[i], &lo[i], mask, data_i);
        }
#endif
      }
      class Eqn : BaseTPP {
        public:
          Eqn() { }
          Eqn(FusedSplitAdamWTPP *p, int eqn_no) : p(p), eqn_no(eqn_no) {
            kernel = (libxsmm_matrix_eqn_function)get_kernel();
            initialized = true;
          }
          void operator()(libxsmm_matrix_eqn_param *eqn_param) {
            if (!initialized) return;
            kernel(eqn_param);
          }
        protected:
          std::string hash_str() override {
            char hash[200];
            snprintf(hash, 200, "fused_split_adamw_eqn%d_t%d_n%d_wd%d", eqn_no, XsmmDtype<T>(), p->N, (p->weight_decay == 0.0 ? 0 : 1));
            return std::string(hash);
          }
          void *build_kernel() override {
            auto in_dt = XsmmDtype<T>();
            libxsmm_blasint ld = p->N;
            auto N = p->N;
            int use_wd = p->weight_decay == 0.0 ? 0 : 1;
            libxsmm_matrix_eqn_function func;
            if (eqn_no == 0) {
              // Equation for exp_avg
              auto my_eqn0 = libxsmm_matrix_eqn_create();
              libxsmm_matrix_eqn_push_back_ternary_op( my_eqn0, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, ld, 2, 0, in_dt ); // avg_i
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32 ); // beta1
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, N, 1, ld, 0, 0, in_dt ); // grad_i
              libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 ); // beta1_1
              libxsmm_matrix_eqn_tree_print( my_eqn0 );
              libxsmm_matrix_eqn_rpn_print( my_eqn0 );
              func = libxsmm_dispatch_matrix_eqn(N, 1, &ld, in_dt, my_eqn0 );
            } else if (eqn_no == 1) {
              // Equation for exp_avg_sq
              auto my_eqn1 = libxsmm_matrix_eqn_create();
              libxsmm_matrix_eqn_push_back_ternary_op( my_eqn1, LIBXSMM_MELTW_TYPE_TERNARY_MULADD, (libxsmm_meltw_ternary_flags)(LIBXSMM_MELTW_FLAG_TERNARY_BCAST_SCALAR_IN_1 | LIBXSMM_MELTW_FLAG_TERNARY_REUSE_IN_2_AS_OUT), LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn1, N, 1, ld, 2, 0, in_dt ); // avg_sq_i
              libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32 ); // beta2
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn1, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn1, LIBXSMM_MELTW_TYPE_UNARY_X2, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn1, N, 1, ld, 0, 0, in_dt ); // grad_i
              libxsmm_matrix_eqn_push_back_arg( my_eqn1, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 ); // beta2_1
              libxsmm_matrix_eqn_tree_print( my_eqn1 );
              libxsmm_matrix_eqn_rpn_print( my_eqn1 );
              func = libxsmm_dispatch_matrix_eqn( N, 1, &ld, in_dt, my_eqn1 );
            } else if (eqn_no == 2) {
              // Equation for data_i (with decay)
              auto my_eqn2 = libxsmm_matrix_eqn_create();
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_UNPACK_TO_BLOCKS, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
              if (use_wd == 1) {
                libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
              }
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_SUB, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_PACK, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn2, N, 1, ld, 4, 0, LIBXSMM_DATATYPE_I16 ); //data_i lo
              libxsmm_matrix_eqn_push_back_arg( my_eqn2, N, 1, ld, 5, 0, LIBXSMM_DATATYPE_I16 ); //data_i hi

              libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_DIV, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn2, N, 1, ld, 2, 0, in_dt ); // avg_i
              libxsmm_matrix_eqn_push_back_binary_op( my_eqn2, LIBXSMM_MELTW_TYPE_BINARY_ADD, LIBXSMM_MELTW_FLAG_BINARY_BCAST_SCALAR_IN_1, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_unary_op( my_eqn2, LIBXSMM_MELTW_TYPE_UNARY_SQRT, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
              libxsmm_matrix_eqn_push_back_arg( my_eqn2, N, 1, ld, 0, 0, in_dt ); // avg_sq_i
              libxsmm_matrix_eqn_push_back_arg( my_eqn2, 1, 1, 1, 1, 0, LIBXSMM_DATATYPE_F32 ); // eps
              libxsmm_matrix_eqn_push_back_arg( my_eqn2, 1, 1, 1, 3, 0, LIBXSMM_DATATYPE_F32 ); // step_size
              if (use_wd == 1) {
                libxsmm_matrix_eqn_push_back_arg( my_eqn2, 1, 1, 1, 6, 0, LIBXSMM_DATATYPE_F32 ); // this scalar is (1-lr*weight_decay)
              }
              libxsmm_matrix_eqn_tree_print( my_eqn2 );
              libxsmm_matrix_eqn_rpn_print( my_eqn2 );
              func = libxsmm_dispatch_matrix_eqn( N, 1, &ld, LIBXSMM_DATATYPE_I16, my_eqn2 );
            } else {
              PCL_ASSERT(false, "Should not come here\n");
            }
            return (void*)func;
          }
        private:
          FusedSplitAdamWTPP *p;
          int eqn_no;
          libxsmm_matrix_eqn_function kernel = NULL;
      };

    private:
      int N = 0;
      float beta1, beta2, weight_decay, eps;
      Eqn eqn0, eqn1, eqn2;
      friend class Eqn;
  };

#if 0
  template<typename T>
    class TPP : public BaseTPP {
      public:
        TPP() { }
        TPP(int N) : N(N){
          kernel = (libxsmm_matrix_eqn_function)get_kernel();
          initialized = true;
        }
        void operator()(T *gout, T *in, T *gin) {
          if (!initialized) return;
          libxsmm_matrix_eqn_param eqn_param;
          libxsmm_matrix_arg arg_array[2];
          arg_array[0].primary = (void*)gout;
          arg_array[1].primary = (void*)in;
          eqn_param.inputs = arg_array;
          eqn_param.output.primary = (void*)gin;

          kernel(&eqn_param);
        }
      protected:
        std::string hash_str() override {
          char hash[200];
          snprintf(hash, 200, "gelu_bwd_eqn_i%d", N);
          return std::string(hash);
        }
        void *build_kernel() override {
          auto dtype = XsmmDtype<T>();
          libxsmm_blasint my_eqn0 = libxsmm_matrix_eqn_create();
          libxsmm_matrix_eqn_push_back_binary_op( my_eqn0, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_MELTW_FLAG_BINARY_NONE, LIBXSMM_DATATYPE_F32 );
          libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, N, N, 0, 0, dtype );
          libxsmm_matrix_eqn_push_back_unary_op( my_eqn0, LIBXSMM_MELTW_TYPE_UNARY_GELU_INV, LIBXSMM_MELTW_FLAG_UNARY_NONE, LIBXSMM_DATATYPE_F32 );
          libxsmm_matrix_eqn_push_back_arg( my_eqn0, 1, N, N, 1, 0, dtype );
          libxsmm_matrix_eqn_tree_print( my_eqn0 );
          libxsmm_matrix_eqn_rpn_print( my_eqn0 );
          return (void*)libxsmm_dispatch_matrix_eqn( M, N, &ld, out_dt, my_eqn0 );
        }
      private:
        int N = 0;
        libxsmm_matrix_eqn_function kernel = NULL;
    };
#endif

}; // namespace pcl

#endif // _XSMM_FUNCTORS_H_
