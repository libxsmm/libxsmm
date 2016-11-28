# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explicitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(strip $(MAKE_PARALLEL)))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
SPLDIR = $(ROOTDIR)/samples
SCRDIR = $(ROOTDIR)/scripts
TSTDIR = $(ROOTDIR)/tests
SRCDIR = $(ROOTDIR)/src
INCDIR = include
BLDDIR = build
OUTDIR = lib
BINDIR = bin
DOCDIR = documentation

# subdirectories for prefix based installation
PINCDIR = $(INCDIR)
POUTDIR = $(OUTDIR)
PBINDIR = $(BINDIR)
PTSTDIR = tests
PDOCDIR = share/libxsmm

# initial default flags
CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -DLIBXSMM_BUILD
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(SRCDIR)

# Python interpreter
PYTHON ?= python

# Version numbers according to interface (version.txt)
VERSION_MAJOR ?= $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py 1)
VERSION_MINOR ?= $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py 2)

# THRESHOLD problem size (M x N x K) determining when to use BLAS; can be zero
THRESHOLD ?= $(shell echo $$((80 * 80 * 80)))

# Generates M,N,K-combinations for each comma separated group e.g., "1, 2, 3" gnerates (1,1,1), (2,2,2),
# and (3,3,3). This way a heterogeneous set can be generated e.g., "1 2, 3" generates (1,1,1), (1,1,2),
# (1,2,1), (1,2,2), (2,1,1), (2,1,2) (2,2,1) out of the first group, and a (3,3,3) for the second group
# To generate a series of square matrices one can specify e.g., make MNK=$(echo $(seq -s, 1 5))
# Alternative to MNK, index sets can be specified separately according to a loop nest relationship
# (M(N(K))) using M, N, and K separately. Please consult the documentation for further details.
MNK ?= 0

# Enable thread-local cache of recently dispatched kernels either
# 0: "disable", 1: "enable", or small power-of-two number.
CACHE ?= 1

# Issue software prefetch instructions (see end of section
# https://github.com/hfp/libxsmm/#generator-driver)
# Use the enumerator 1...10, or the exact strategy
# name pfsigonly...AL2_BL2viaC_CL2.
#  1: auto-select
#  2: pfsigonly
#  3: BL2viaC
#  4: curAL2
#  7: curAL2_BL2viaC
#  5: AL2
#  6: AL2_BL2viaC
#  8: AL2jpst
#  9: AL2jpst_BL2viaC
# 10: AL2_BL2viaC_CL2
PREFETCH ?= 1

# Preferred precision when registering statically generated code versions
# 0: SP and DP code versions to be registered
# 1: SP only
# 2: DP only
PRECISION ?= 0

# Support SMM kernels with larger extent(s)
# 0: optimized JIT descriptor size
# 1: regular descriptor size
BIG ?= 1
ifeq (0,$(BIG))
  DFLAGS += -DLIBXSMM_GENERATOR_SMALLDESC
endif

# Specify an alignment (Bytes)
ALIGNMENT ?= 64

# Generate code using aligned Load/Store instructions
# !=0: enable if lda/ldc (m) is a multiple of ALIGNMENT
# ==0: disable emitting aligned Load/Store instructions
ALIGNED_STORES ?= 0
ALIGNED_LOADS ?= 0

# Alpha argument of GEMM
# Supported: 1.0
ALPHA ?= 1
ifneq (1,$(ALPHA))
  $(error ALPHA needs to be 1)
endif

# Beta argument of GEMM
# Supported: 0.0, 1.0
# 0: C  = A * B
# 1: C += A * B
BETA ?= 1
ifneq (1,$(BETA))
ifneq (0,$(BETA))
  $(error BETA needs to be either 0 or 1)
endif
endif

# Determines if the library is thread-safe
THREADS ?= 1

# 0: produces shared library files suitable for dynamic linkage
# 1: produces library archives suitable for static linkage
STATIC ?= 1

# Determines if the library can act as a wrapper-library (GEMM)
# 1: enables wrapping SGEMM and DGEMM
# 2: enables wrapping DGEMM only
WRAP ?= 0

# JIT backend is enabled by default
JIT ?= 1
ifneq (0,$(JIT))
  AVX ?= 0
  SSE ?= 1
endif

# Profiling JIT code using Linux Perf
# PERF=0: disabled (default)
# PERF=1: enabled (without JITDUMP)
# PERF=2: enabled (with JITDUMP)
#
# Additional support for jitdump
# JITDUMP=0: disabled (default)
# JITDUMP=1: enabled
# PERF=2: enabled
#
ifneq (,$(PERF))
ifneq (0,$(PERF))
ifneq (1,$(PERF))
  JITDUMP ?= 1
endif
endif
endif
JITDUMP ?= 0

ifneq (0,$(JITDUMP))
  PERF ?= 1
endif

PERF ?= 0
ifneq (0,$(PERF))
  SYM ?= 1
endif

# OpenMP is disabled by default and LIBXSMM is
# always agnostic wrt the threading runtime
OMP ?= 0

ifneq (,$(MKL))
  BLAS = $(MKL)
endif

BLAS_WARNING ?= 0
ifeq (0,$(STATIC))
  ifeq (Windows_NT,$(OS)) # !UNAME
    BLAS_WARNING = 1
    BLAS ?= 2
  else ifeq (Darwin,$(shell uname))
    BLAS_WARNING = 1
    BLAS ?= 2
  endif
endif

ifneq (1,$(CACHE))
  DFLAGS += -DLIBXSMM_CACHESIZE=$(CACHE)
endif

# state to be excluded from tracking the (re-)build state
EXCLUDE_STATE = BLAS_WARNING PREFIX

# include common Makefile artifacts
include $(ROOTDIR)/Makefile.inc

ifeq (1,$(AVX))
  GENTARGET = snb
else ifeq (2,$(AVX))
  GENTARGET = hsw
else ifeq (3,$(AVX))
  GENTARGET = knl
else ifneq (0,$(SSE))
  GENTARGET = wsm
else
  GENTARGET = noarch
endif

ifeq (0,$(STATIC))
  GENERATOR = @$(ENV) \
    LD_LIBRARY_PATH=$(OUTDIR):$${LD_LIBRARY_PATH} \
    PATH=$(OUTDIR):$${PATH} \
  $(BINDIR)/libxsmm_gemm_generator
else
  GENERATOR = $(BINDIR)/libxsmm_gemm_generator
endif

INDICES ?= $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py -1 $(THRESHOLD) $(words $(MNK)) $(MNK) $(words $(M)) $(words $(N)) $(M) $(N) $(K))
NINDICES = $(words $(INDICES))

HEADERS = $(shell ls -1 $(SRCDIR)/*.h 2> /dev/null | tr "\n" " ") \
          $(shell ls -1 $(SRCDIR)/template/*.c 2> /dev/null | tr "\n" " ") \
          $(SRCDIR)/libxsmm_gemm_diff.c $(SRCDIR)/libxsmm_hash.c \
          $(ROOTDIR)/include/libxsmm_dnn.h \
          $(ROOTDIR)/include/libxsmm_cpuid.h \
          $(ROOTDIR)/include/libxsmm_frontend.h \
          $(ROOTDIR)/include/libxsmm_generator.h \
          $(ROOTDIR)/include/libxsmm_intrinsics_x86.h \
          $(ROOTDIR)/include/libxsmm_macros.h \
          $(ROOTDIR)/include/libxsmm_malloc.h \
          $(ROOTDIR)/include/libxsmm_spmdm.h \
          $(ROOTDIR)/include/libxsmm_sync.h \
          $(ROOTDIR)/include/libxsmm_timer.h \
          $(ROOTDIR)/include/libxsmm_typedefs.h

SRCFILES_KERNELS = $(patsubst %,$(BLDDIR)/mm_%.c,$(INDICES))
SRCFILES_GEN_LIB = $(patsubst %,$(SRCDIR)/%,$(wildcard $(SRCDIR)/generator_*.c) \
                   libxsmm_cpuid_x86.c libxsmm_malloc.c libxsmm_sync.c \
                   libxsmm_timer.c libxsmm_trace.c libxsmm_perf.c)
SRCFILES_GEN_GEMM_BIN = $(patsubst %,$(SRCDIR)/%,libxsmm_generator_gemm_driver.c)
SRCFILES_GEN_CONV_BIN = $(patsubst %,$(SRCDIR)/%,libxsmm_generator_convolution_driver.c)
OBJFILES_GEN_GEMM_BIN = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_GEMM_BIN))))
OBJFILES_GEN_CONV_BIN = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_CONV_BIN))))
OBJFILES_GEN_LIB = $(patsubst %,$(BLDDIR)/intel64/%.o,$(basename $(notdir $(SRCFILES_GEN_LIB))))
OBJFILES_HST = $(BLDDIR)/intel64/libxsmm_main.o $(BLDDIR)/intel64/libxsmm_dump.o \
               $(BLDDIR)/intel64/libxsmm_gemm.o $(BLDDIR)/intel64/libxsmm_trans.o \
               $(BLDDIR)/intel64/libxsmm_spmdm.o \
               $(BLDDIR)/intel64/libxsmm_dnn.o $(BLDDIR)/intel64/libxsmm_dnn_handle.o \
               $(BLDDIR)/intel64/libxsmm_dnn_convolution_forward.o \
               $(BLDDIR)/intel64/libxsmm_dnn_convolution_backward.o \
               $(BLDDIR)/intel64/libxsmm_dnn_convolution_weight_update.o
OBJFILES_MIC = $(BLDDIR)/mic/libxsmm_main.o $(BLDDIR)/mic/libxsmm_dump.o \
               $(BLDDIR)/mic/libxsmm_gemm.o $(BLDDIR)/mic/libxsmm_trans.o \
               $(BLDDIR)/mic/libxsmm_dnn.o $(BLDDIR)/mic/libxsmm_dnn_handle.o \
               $(BLDDIR)/mic/libxsmm_dnn_convolution_forward.o \
               $(BLDDIR)/mic/libxsmm_dnn_convolution_backward.o \
               $(BLDDIR)/mic/libxsmm_dnn_convolution_weight_update.o \
               $(BLDDIR)/mic/libxsmm_cpuid_x86.o $(BLDDIR)/mic/libxsmm_malloc.o \
               $(BLDDIR)/mic/libxsmm_sync.o $(BLDDIR)/mic/libxsmm_timer.o \
               $(BLDDIR)/mic/libxsmm_trace.o $(BLDDIR)/mic/libxsmm_perf.o
KRNOBJS_HST  = $(patsubst %,$(BLDDIR)/intel64/mm_%.o,$(INDICES))
KRNOBJS_MIC  = $(patsubst %,$(BLDDIR)/mic/mm_%.o,$(INDICES))
EXTOBJS_HST  = $(BLDDIR)/intel64/libxsmm_ext.o \
               $(BLDDIR)/intel64/libxsmm_ext_gemm.o $(BLDDIR)/intel64/libxsmm_ext_trans.o
EXTOBJS_MIC  = $(BLDDIR)/mic/libxsmm_ext.o \
               $(BLDDIR)/mic/libxsmm_ext_gemm.o $(BLDDIR)/mic/libxsmm_ext_trans.o
NOBLAS_HST   = $(BLDDIR)/intel64/libxsmm_noblas.o
NOBLAS_MIC   = $(BLDDIR)/mic/libxsmm_noblas.o

# list of object might be "incomplete" if not all code gen. FLAGS are supplied with clean target!
OBJECTS = $(OBJFILES_GEN_LIB) $(OBJFILES_GEN_GEMM_BIN) $(OBJFILES_GEN_CONV_BIN) $(OBJFILES_HST) $(OBJFILES_MIC) \
          $(KRNOBJS_HST) $(KRNOBJS_MIC) $(EXTOBJS_HST) $(EXTOBJS_MIC) $(NOBLAS_HST) $(NOBLAS_MIC)
ifneq (,$(strip $(FC)))
  FTNOBJS = $(BLDDIR)/intel64/libxsmm-mod.o $(BLDDIR)/mic/libxsmm-mod.o
endif

.PHONY: libxsmm
libxsmm: lib generator

.PHONY: lib
lib: headers drytest lib_hst lib_mic

.PHONY: all
all: libxsmm samples

.PHONY: headers
headers: cheader cheader_only fheader

.PHONY: interface
interface: headers module

.PHONY: lib_mic
lib_mic: clib_mic flib_mic ext_mic noblas_mic

.PHONY: lib_hst
lib_hst: clib_hst flib_hst ext_hst noblas_hst

PREFETCH_UID = 0
PREFETCH_SCHEME = nopf
PREFETCH_TYPE = 0

ifneq (0,$(shell echo $$((2 <= $(PREFETCH) && $(PREFETCH) <= 10))))
  PREFETCH_UID = $(PREFETCH)
else ifeq (1,$(PREFETCH)) # auto
  PREFETCH_UID = 1
else ifeq (pfsigonly,$(PREFETCH))
  PREFETCH_UID = 2
else ifeq (BL2viaC,$(PREFETCH))
  PREFETCH_UID = 3
else ifeq (curAL2,$(PREFETCH))
  PREFETCH_UID = 4
else ifeq (curAL2_BL2viaC,$(PREFETCH))
  PREFETCH_UID = 5
else ifeq (AL2,$(PREFETCH))
  PREFETCH_UID = 6
else ifeq (AL2_BL2viaC,$(PREFETCH))
  PREFETCH_UID = 7
else ifeq (AL2jpst,$(PREFETCH))
  PREFETCH_UID = 8
else ifeq (AL2jpst_BL2viaC,$(PREFETCH))
  PREFETCH_UID = 9
else ifeq (AL2_BL2viaC_CL2,$(PREFETCH))
  PREFETCH_UID = 10
endif

# Mapping build options to libxsmm_gemm_prefetch_type (see include/libxsmm_typedefs.h)
ifeq (1,$(PREFETCH_UID))
  # Prefetch "auto" is a pseudo-strategy introduced by the frontend;
  # select "pfsigonly" for statically generated code.
  PREFETCH_SCHEME = pfsigonly
  PREFETCH_TYPE = -1
  ifneq (0,$(MIC))
    ifneq (0,$(MPSS))
      PREFETCH_SCHEME_MIC = AL2_BL2viaC_CL2
    endif
  endif
else ifeq (2,$(PREFETCH_UID))
  PREFETCH_SCHEME = pfsigonly
  PREFETCH_TYPE = 1
else ifeq (3,$(PREFETCH_UID))
  PREFETCH_SCHEME = BL2viaC
  PREFETCH_TYPE = 8
else ifeq (4,$(PREFETCH_UID))
  PREFETCH_SCHEME = curAL2
  PREFETCH_TYPE = 16
else ifeq (5,$(PREFETCH_UID))
  PREFETCH_SCHEME = curAL2_BL2viaC
  PREFETCH_TYPE = $(shell echo $$((8 | 16)))
else ifeq (6,$(PREFETCH_UID))
  PREFETCH_SCHEME = AL2
  PREFETCH_TYPE = 2
else ifeq (7,$(PREFETCH_UID))
  PREFETCH_SCHEME = AL2_BL2viaC
  PREFETCH_TYPE = $(shell echo $$((8 | 2)))
else ifeq (8,$(PREFETCH_UID))
  PREFETCH_SCHEME = AL2jpst
  PREFETCH_TYPE = 4
else ifeq (9,$(PREFETCH_UID))
  PREFETCH_SCHEME = AL2jpst_BL2viaC
  PREFETCH_TYPE = $(shell echo $$((8 | 4)))
else ifeq (10,$(PREFETCH_UID))
  PREFETCH_SCHEME = AL2_BL2viaC_CL2
  PREFETCH_TYPE = $(shell echo $$((8 | 2 | 32)))
endif
ifeq (,$(PREFETCH_SCHEME_MIC))
  PREFETCH_SCHEME_MIC = $(PREFETCH_SCHEME)
endif

# Mapping build options to libxsmm_gemm_flags (see include/libxsmm_typedefs.h)
FLAGS = $(shell echo $$((((0!=$(ALIGNED_LOADS))*4) | ((0!=$(ALIGNED_STORES))*8))))

SUPPRESS_UNUSED_VARIABLE_WARNINGS = LIBXSMM_UNUSED(A); LIBXSMM_UNUSED(B); LIBXSMM_UNUSED(C);
ifneq (nopf,$(PREFETCH_SCHEME))
  #SUPPRESS_UNUSED_VARIABLE_WARNINGS += LIBXSMM_UNUSED(A_prefetch); LIBXSMM_UNUSED(B_prefetch);
  #SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXSMM_UNUSED(C_prefetch);~
  SUPPRESS_UNUSED_PREFETCH_WARNINGS = $(NULL)  LIBXSMM_UNUSED(A_prefetch); LIBXSMM_UNUSED(B_prefetch); LIBXSMM_UNUSED(C_prefetch);~
endif

.PHONY: config
config: $(INCDIR)/libxsmm_config.h
$(INCDIR)/libxsmm_config.h: $(INCDIR)/.make .state $(SRCDIR)/template/libxsmm_config.h \
                            $(SCRDIR)/libxsmm_config.py $(SCRDIR)/libxsmm_utilities.py \
                            $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@if [ -e $(ROOTDIR)/.hooks/install.sh ]; then \
		$(ROOTDIR)/.hooks/install.sh; \
	fi
	@cp $(ROOTDIR)/include/libxsmm_dnn.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_cpuid.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_frontend.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_generator.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_intrinsics_x86.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_macros.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_malloc.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_spmdm.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_sync.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_timer.h $(INCDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/include/libxsmm_typedefs.h $(INCDIR) 2> /dev/null || true
	@$(PYTHON) $(SCRDIR)/libxsmm_config.py $(SRCDIR)/template/libxsmm_config.h \
		$(MAKE_ILP64) $(OFFLOAD) $(ALIGNMENT) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(shell echo $$(($(THREADS)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) > $@
	$(info ================================================================================)
	$(info LIBXSMM $(shell $(PYTHON) $(SCRDIR)/libxsmm_utilities.py))
	$(info --------------------------------------------------------------------------------)
	$(info $(GINFO))
	$(info $(CINFO))
ifneq (,$(strip $(FC)))
	$(info $(FINFO))
endif
	$(info --------------------------------------------------------------------------------)
ifeq (,$(strip $(FC)))
ifeq (,$(strip $(FC_VERSION_STRING)))
	$(info Fortran Compiler is missing: building without Fortran support!)
else
	$(info Fortran Compiler $(FC_VERSION_STRING) is outdated!)
endif
	$(info ================================================================================)
endif
ifeq (0,$(STATIC))
ifeq (Windows_NT,$(UNAME))
	$(info The shared link-time wrapper (libxsmmext) is not supported under Windows/Cygwin!)
	$(info ================================================================================)
endif
endif
ifneq (0,$(BLAS_WARNING))
	$(info Building a shared library requires to link against BLAS since there is)
	$(info no runtime resolution/search for weak symbols implemented for this OS.)
endif
ifneq (0,$(BLAS))
ifeq (0,$(BLAS_WARNING))
	$(info LIBXSMM is link-time agnostic with respect to BLAS/GEMM!)
	$(info Linking a certain BLAS library may prevent users to decide.)
endif
ifeq (1,$(BLAS))
	$(info LIBXSMM's THRESHOLD already prevents calling small GEMMs!)
	$(info A sequential BLAS is superfluous with respect to LIBXSMM.)
endif
	$(info ================================================================================)
else ifneq (0,$(NOBLAS))
	$(info LIBXSMM's link-time BLAS dependency is removed (fallback might be unavailable!))
	$(info ================================================================================)
endif

.PHONY: cheader
cheader: $(INCDIR)/libxsmm.h
$(INCDIR)/libxsmm.h: $(SCRDIR)/libxsmm_interface.py \
                     $(SRCDIR)/template/libxsmm.h $(ROOTDIR)/version.txt \
                     $(INCDIR)/libxsmm_config.h $(HEADERS)
	@$(PYTHON) $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/template/libxsmm.h \
		$(PRECISION) $(MAKE_ILP64) $(PREFETCH_TYPE) $(INDICES) > $@

.PHONY: cheader_only
cheader_only: $(INCDIR)/libxsmm_source.h
$(INCDIR)/libxsmm_source.h: $(INCDIR)/.make $(SCRDIR)/libxsmm_source.sh $(INCDIR)/libxsmm.h
	@$(SCRDIR)/libxsmm_source.sh > $@

.PHONY: fheader
fheader: $(INCDIR)/libxsmm.f
$(INCDIR)/libxsmm.f: $(BLDDIR)/.make \
                     $(ROOTDIR)/version.txt $(INCDIR)/libxsmm_config.h \
                     $(SRCDIR)/template/libxsmm.f $(SCRDIR)/libxsmm_interface.py \
                     $(ROOTDIR)/Makefile $(ROOTDIR)/Makefile.inc
	@$(PYTHON) $(SCRDIR)/libxsmm_interface.py $(SRCDIR)/template/libxsmm.f \
		$(PRECISION) $(MAKE_ILP64) $(PREFETCH_TYPE) $(INDICES) | \
	$(PYTHON) $(SCRDIR)/libxsmm_config.py /dev/stdin \
		$(MAKE_ILP64) $(OFFLOAD) $(ALIGNMENT) $(PREFETCH_TYPE) \
		$(shell echo $$((0<$(THRESHOLD)?$(THRESHOLD):0))) \
		$(shell echo $$(($(THREADS)+$(OMP)))) \
		$(JIT) $(FLAGS) $(ALPHA) $(BETA) $(INDICES) | \
	sed "/ATTRIBUTES OFFLOAD:MIC/d" > $@

.PHONY: sources
sources: $(SRCFILES_KERNELS) $(BLDDIR)/libxsmm_dispatch.h
$(BLDDIR)/libxsmm_dispatch.h: $(BLDDIR)/.make $(SCRDIR)/libxsmm_dispatch.py $(SRCFILES_KERNELS) \
                              $(INCDIR)/libxsmm.h
	@$(PYTHON) $(SCRDIR)/libxsmm_dispatch.py $(PRECISION) $(THRESHOLD) $(INDICES) > $@

$(BLDDIR)/%.c: $(BLDDIR)/.make $(INCDIR)/libxsmm.h $(BINDIR)/libxsmm_gemm_generator $(SCRDIR)/libxsmm_utilities.py $(SCRDIR)/libxsmm_specialized.py
ifneq (,$(strip $(SRCFILES_KERNELS)))
	$(eval MVALUE := $(shell echo $(basename $@) | cut -d_ -f2))
	$(eval NVALUE := $(shell echo $(basename $@) | cut -d_ -f3))
	$(eval KVALUE := $(shell echo $(basename $@) | cut -d_ -f4))
	$(eval MNVALUE := $(MVALUE))
	$(eval NMVALUE := $(NVALUE))
	$(eval ASTSP := $(shell echo $$((0!=$(ALIGNED_STORES)&&0==($(MNVALUE)*4)%$(ALIGNMENT)))))
	$(eval ASTDP := $(shell echo $$((0!=$(ALIGNED_STORES)&&0==($(MNVALUE)*8)%$(ALIGNMENT)))))
	$(eval ALDSP := $(shell echo $$((0!=$(ALIGNED_LOADS)&&0==($(MNVALUE)*4)%$(ALIGNMENT)))))
	$(eval ALDDP := $(shell echo $$((0!=$(ALIGNED_LOADS)&&0==($(MNVALUE)*8)%$(ALIGNMENT)))))
	@echo "#include <libxsmm.h>" > $@
	@echo >> $@
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knc_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knc_dp" >> $@
endif
endif
endif
ifeq (noarch,$(GENTARGET))
ifneq (,$(CTARGET))
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knl_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_sp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_knl_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_hsw_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_snb_dp" >> $@
	@echo "#define LIBXSMM_GENTARGET_wsm_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) knl $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) hsw $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) snb $(PREFETCH_SCHEME) SP
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) wsm $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_knl $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) knl $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_hsw $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) hsw $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_snb $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) snb $(PREFETCH_SCHEME) DP
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_wsm $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) wsm $(PREFETCH_SCHEME) DP
endif
endif # target
else # noarch
ifneq (2,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_sp" >> $@
endif
ifneq (1,$(PRECISION))
	@echo "#define LIBXSMM_GENTARGET_$(GENTARGET)_dp" >> $@
endif
	@echo >> $@
	@echo >> $@
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTSP) $(GENTARGET) $(PREFETCH_SCHEME) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_$(GENTARGET) $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDDP) $(ASTDP) $(GENTARGET) $(PREFETCH_SCHEME) DP
endif
endif # noarch
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (2,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_s$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTDP) knc $(PREFETCH_SCHEME_MIC) SP
endif
ifneq (1,$(PRECISION))
	$(GENERATOR) dense $@ libxsmm_d$(basename $(notdir $@))_knc $(MNVALUE) $(NMVALUE) $(KVALUE) $(MNVALUE) $(KVALUE) $(MNVALUE) $(ALPHA) $(BETA) $(ALDSP) $(ASTDP) knc $(PREFETCH_SCHEME_MIC) DP
endif
endif
endif
	$(eval TMPFILE = $(shell mktemp /tmp/fileXXXXXX))
	@cat $@ | sed \
		-e "s/void libxsmm_/LIBXSMM_INLINE LIBXSMM_RETARGETABLE void libxsmm_/" \
		-e "s/#ifndef NDEBUG/$(SUPPRESS_UNUSED_PREFETCH_WARNINGS)#ifdef LIBXSMM_NEVER_DEFINED/" \
		-e "s/#pragma message (\".*KERNEL COMPILATION ERROR in: \" __FILE__)/  $(SUPPRESS_UNUSED_VARIABLE_WARNINGS)/" \
		-e "/#error No kernel was compiled, lacking support for current architecture?/d" \
		-e "/#pragma message (\".*KERNEL COMPILATION WARNING: compiling ..* code on ..* or newer architecture: \" __FILE__)/d" \
		| tr "~" "\n" > $(TMPFILE)
	@$(PYTHON) $(SCRDIR)/libxsmm_specialized.py $(PRECISION) $(MVALUE) $(NVALUE) $(KVALUE) $(PREFETCH_TYPE) >> $(TMPFILE)
	@mv $(TMPFILE) $@
endif

ifneq (0,$(JIT))
ifneq (0,$(SYM))
ifeq (,$(filter Darwin,$(UNAME)))
  ifneq (0,$(PERF))
    DFLAGS += -DLIBXSMM_PERF
    ifneq (0,$(JITDUMP))
      DFLAGS += -DLIBXSMM_PERF_JITDUMP
    endif
  endif

  VTUNEROOT = $(shell env | grep VTUNE_AMPLIFIER | grep -m1 _DIR | cut -d= -f2-)
  ifneq (,$(wildcard $(VTUNEROOT)/lib64/libjitprofiling.$(SLIBEXT)))
    LIBJITPROFILING = $(BLDDIR)/jitprofiling/libjitprofiling.$(SLIBEXT)
    OBJJITPROFILING = $(BLDDIR)/jitprofiling/*.o
    DFLAGS += -DLIBXSMM_VTUNE
    IFLAGS += -I$(VTUNEROOT)/include
    ifneq (0,$(INTEL))
      CXXFLAGS += -diag-disable 271
      CFLAGS += -diag-disable 271
    endif
$(LIBJITPROFILING): $(BLDDIR)/jitprofiling/.make
	@cp $(VTUNEROOT)/lib64/libjitprofiling.$(SLIBEXT) $(BLDDIR)/jitprofiling
	@cd $(BLDDIR)/jitprofiling; $(AR) x libjitprofiling.$(SLIBEXT)
  else
.PHONY: $(LIBJITPROFILING)
  endif
endif
endif
endif

define DEFINE_COMPILE_RULE
$(1): $(2) $(3) $(dir $(1))/.make
	$(CC) $(4) -c $(2) -o $(1)
endef

EXTCFLAGS = -DLIBXSMM_BUILD_EXT
ifneq (0,$(STATIC))
ifneq (0,$(WRAP))
  EXTCFLAGS += -DLIBXSMM_GEMM_WRAP=$(WRAP)
endif
endif

ifeq (0,$(OMP))
ifeq (,$(filter environment% override command%,$(origin OMP)))
  EXTCFLAGS += $(OMPFLAG)
  EXTLDFLAGS += $(OMPFLAG)
endif
endif

ifneq (0,$(MIC))
ifneq (0,$(MPSS))
$(foreach OBJ,$(OBJFILES_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h, \
  $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic)))
$(foreach OBJ,$(KRNOBJS_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(BLDDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  -mmic $(CSTD) $(CPEDANTIC) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(EXTOBJS_MIC),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ), $(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  -mmic $(EXTCFLAGS) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(eval $(call DEFINE_COMPILE_RULE,$(NOBLAS_MIC),$(SRCDIR)/libxsmm_ext.c,$(INCDIR)/libxsmm.h, \
  -mmic $(NOBLAS_CFLAGS) $(NOBLAS_DFLAGS) $(NOBLAS_IFLAGS) $(DNOBLAS)))
endif
endif

$(foreach OBJ,$(OBJFILES_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h, \
  $(CTARGET) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(KRNOBJS_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(BLDDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(CTARGET) $(CSTD) $(CPEDANTIC) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(EXTOBJS_HST),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(CTARGET) $(EXTCFLAGS) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_LIB),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(CSTD) $(CPEDANTIC) $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_GEMM_BIN),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(foreach OBJ,$(OBJFILES_GEN_CONV_BIN),$(eval $(call DEFINE_COMPILE_RULE, \
  $(OBJ),$(patsubst %.o,$(SRCDIR)/%.c,$(notdir $(OBJ))), \
  $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h, \
  $(CFLAGS) $(DFLAGS) $(IFLAGS))))
$(eval $(call DEFINE_COMPILE_RULE,$(NOBLAS_HST),$(SRCDIR)/libxsmm_ext.c,$(INCDIR)/libxsmm.h, \
  $(CTARGET) $(NOBLAS_CFLAGS) $(NOBLAS_DFLAGS) $(NOBLAS_IFLAGS) $(DNOBLAS)))

.PHONY: compile_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
compile_mic:
$(BLDDIR)/mic/%.o: $(BLDDIR)/%.c $(BLDDIR)/mic/.make $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
endif
endif

.PHONY: compile_hst
compile_hst:
$(BLDDIR)/intel64/%.o: $(BLDDIR)/%.c $(BLDDIR)/intel64/.make $(INCDIR)/libxsmm.h $(INCDIR)/libxsmm_source.h $(BLDDIR)/libxsmm_dispatch.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

.PHONY: module_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
module_mic: $(INCDIR)/mic/libxsmm.mod
$(BLDDIR)/mic/libxsmm-mod.o: $(BLDDIR)/mic/.make $(INCDIR)/mic/.make $(INCDIR)/libxsmm.f
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $(INCDIR)/libxsmm.f -o $@ $(FMFLAGS) $(INCDIR)/mic
$(INCDIR)/mic/libxsmm.mod: $(BLDDIR)/mic/libxsmm-mod.o
	@if [ -e $(BLDDIR)/mic/libxsmm.mod ]; then cp $(BLDDIR)/mic/libxsmm.mod $@; fi
	@if [ -e $(BLDDIR)/mic/LIBXSMM.mod ]; then cp $(BLDDIR)/mic/LIBXSMM.mod $@; fi
	@touch $@
else
.PHONY: $(BLDDIR)/mic/libxsmm-mod.o
.PHONY: $(INCDIR)/mic/libxsmm.mod
endif
else
.PHONY: $(BLDDIR)/mic/libxsmm-mod.o
.PHONY: $(INCDIR)/mic/libxsmm.mod
endif
else
.PHONY: $(BLDDIR)/mic/libxsmm-mod.o
.PHONY: $(INCDIR)/mic/libxsmm.mod
endif

.PHONY: module_hst
ifneq (,$(strip $(FC)))
module_hst: $(INCDIR)/libxsmm.mod
$(BLDDIR)/intel64/libxsmm-mod.o: $(BLDDIR)/intel64/.make $(INCDIR)/libxsmm.f
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $(INCDIR)/libxsmm.f -o $@ $(FMFLAGS) $(INCDIR)
$(INCDIR)/libxsmm.mod: $(BLDDIR)/intel64/libxsmm-mod.o
	@if [ -e $(BLDDIR)/intel64/libxsmm.mod ]; then cp $(BLDDIR)/intel64/libxsmm.mod $@; fi
	@if [ -e $(BLDDIR)/intel64/LIBXSMM.mod ]; then cp $(BLDDIR)/intel64/LIBXSMM.mod $@; fi
	@touch $@
else
.PHONY: $(BLDDIR)/intel64/libxsmm-mod.o
.PHONY: $(INCDIR)/libxsmm.mod
endif

.PHONY: module
module: module_hst module_mic

.PHONY: build_generator_lib
build_generator_lib: $(OUTDIR)/libxsmmgen.$(LIBEXT)
$(OUTDIR)/libxsmmgen.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_GEN_LIB)
ifeq (0,$(STATIC))
ifneq (Darwin,$(UNAME))
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(call soname,$@ $(VERSION_MAJOR)) $(OBJFILES_GEN_LIB) $(LDFLAGS) $(CLDFLAGS) -lrt
else
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(call soname,$@ $(VERSION_MAJOR)) $(OBJFILES_GEN_LIB) $(LDFLAGS) $(CLDFLAGS)
endif
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else
	$(AR) -rs $@ $(OBJFILES_GEN_LIB)
endif

.PHONY: generator
generator: $(BINDIR)/libxsmm_gemm_generator $(BINDIR)/libxsmm_conv_generator
$(BINDIR)/libxsmm_gemm_generator: $(BINDIR)/.make $(OBJFILES_GEN_GEMM_BIN) $(OUTDIR)/libxsmmgen.$(LIBEXT)
	$(CC) -o $@ $(OBJFILES_GEN_GEMM_BIN) $(call abslib,$(OUTDIR)/libxsmmgen.$(LIBEXT)) $(LDFLAGS) $(CLDFLAGS)
$(BINDIR)/libxsmm_conv_generator: $(BINDIR)/.make $(OBJFILES_GEN_CONV_BIN) $(OUTDIR)/libxsmmgen.$(LIBEXT)
	$(CC) -o $@ $(OBJFILES_GEN_CONV_BIN) $(call abslib,$(OUTDIR)/libxsmmgen.$(LIBEXT)) $(LDFLAGS) $(CLDFLAGS)

.PHONY: clib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
clib_mic: $(OUTDIR)/mic/libxsmm.$(LIBEXT)
$(OUTDIR)/mic/libxsmm.$(LIBEXT): $(OUTDIR)/mic/.make $(OBJFILES_MIC) $(KRNOBJS_MIC)
ifeq (0,$(STATIC))
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -mmic -shared $(call soname,$@ $(VERSION_MAJOR)) $(OBJFILES_MIC) $(KRNOBJS_MIC) $(LDFLAGS) $(CLDFLAGS)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else
	$(AR) -rs $@ $(OBJFILES_MIC) $(KRNOBJS_MIC)
endif
endif
endif

.PHONY: clib_hst
clib_hst: $(OUTDIR)/libxsmm.$(LIBEXT)
$(OUTDIR)/libxsmm.$(LIBEXT): $(OUTDIR)/.make $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(LIBJITPROFILING)
ifeq (0,$(STATIC))
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(call soname,$@ $(VERSION_MAJOR)) $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(LIBJITPROFILING) $(LDFLAGS) $(CLDFLAGS)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else
	$(AR) -rs $@ $(OBJFILES_HST) $(OBJFILES_GEN_LIB) $(KRNOBJS_HST) $(OBJJITPROFILING)
endif

.PHONY: flib_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ifneq (,$(strip $(FC)))
flib_mic: $(OUTDIR)/mic/libxsmmf.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/mic/libxsmmf.$(LIBEXT): $(INCDIR)/mic/libxsmm.mod $(OUTDIR)/mic/libxsmm.$(LIBEXT)
	$(FC) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -mmic -shared $(FCMTFLAGS) $(call soname,$@ $(VERSION_MAJOR)) $(BLDDIR)/mic/libxsmm-mod.o $(call abslib,$(OUTDIR)/mic/libxsmm.$(LIBEXT)) $(LDFLAGS) $(FLDFLAGS)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else
$(OUTDIR)/mic/libxsmmf.$(LIBEXT): $(INCDIR)/mic/libxsmm.mod $(OUTDIR)/mic/.make
	$(AR) -rs $@ $(BLDDIR)/mic/libxsmm-mod.o
endif
else
.PHONY: $(OUTDIR)/mic/libxsmmf.$(LIBEXT)
endif
endif
endif

.PHONY: flib_hst
ifneq (,$(strip $(FC)))
flib_hst: $(OUTDIR)/libxsmmf.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/libxsmmf.$(LIBEXT): $(INCDIR)/libxsmm.mod $(OUTDIR)/libxsmm.$(LIBEXT)
	$(FC) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(FCMTFLAGS) $(call soname,$@ $(VERSION_MAJOR)) $(BLDDIR)/intel64/libxsmm-mod.o $(call abslib,$(OUTDIR)/libxsmm.$(LIBEXT)) $(LDFLAGS) $(FLDFLAGS)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else
$(OUTDIR)/libxsmmf.$(LIBEXT): $(INCDIR)/libxsmm.mod $(OUTDIR)/.make
	$(AR) -rs $@ $(BLDDIR)/intel64/libxsmm-mod.o
endif
else
.PHONY: $(OUTDIR)/libxsmmf.$(LIBEXT)
endif

.PHONY: ext_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
ext_mic: $(OUTDIR)/mic/libxsmmext.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/mic/libxsmmext.$(LIBEXT): $(OUTDIR)/mic/.make $(EXTOBJS_MIC) $(OUTDIR)/mic/libxsmm.$(DLIBEXT)
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -mmic -shared $(EXTLDFLAGS) $(call soname,$@ $(VERSION_MAJOR)) $(EXTOBJS_MIC) $(call abslib,$(OUTDIR)/mic/libxsmm.$(DLIBEXT)) $(LDFLAGS) $(CLDFLAGS)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else
$(OUTDIR)/mic/libxsmmext.$(LIBEXT): $(OUTDIR)/mic/.make $(EXTOBJS_MIC)
	$(AR) -rs $@ $(EXTOBJS_MIC)
endif
endif
endif

.PHONY: ext_hst
ext_hst: $(OUTDIR)/libxsmmext.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/libxsmmext.$(LIBEXT): $(OUTDIR)/.make $(EXTOBJS_HST) $(OUTDIR)/libxsmm.$(DLIBEXT)
ifeq (Darwin,$(UNAME))
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(call soname,$@ $(VERSION_MAJOR)) $(EXTOBJS_HST) $(call abslib,$(OUTDIR)/libxsmm.$(DLIBEXT)) $(LDFLAGS) $(CLDFLAGS)
else
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(EXTLDFLAGS) $(call soname,$@ $(VERSION_MAJOR)) $(EXTOBJS_HST) $(call abslib,$(OUTDIR)/libxsmm.$(DLIBEXT)) $(LDFLAGS) $(CLDFLAGS)
endif
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else # static
$(OUTDIR)/libxsmmext.$(LIBEXT): $(OUTDIR)/.make $(EXTOBJS_HST)
	$(AR) -rs $@ $(EXTOBJS_HST)
endif

.PHONY: noblas_mic
ifneq (0,$(MIC))
ifneq (0,$(MPSS))
noblas_mic: $(OUTDIR)/mic/libxsmmnoblas.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/mic/libxsmmnoblas.$(LIBEXT): $(OUTDIR)/mic/.make $(NOBLAS_MIC)
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -mmic -shared $(EXTLDFLAGS) $(call soname,$@ $(VERSION_MAJOR)) $(NOBLAS_MIC) $(LDFLAGS) $(CLDFLAGS)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else
$(OUTDIR)/mic/libxsmmnoblas.$(LIBEXT): $(OUTDIR)/mic/.make $(NOBLAS_MIC)
	$(AR) -rs $@ $(NOBLAS_MIC)
endif
endif
endif

.PHONY: noblas_hst
noblas_hst: $(OUTDIR)/libxsmmnoblas.$(LIBEXT)
ifeq (0,$(STATIC))
$(OUTDIR)/libxsmmnoblas.$(LIBEXT): $(OUTDIR)/.make $(NOBLAS_HST)
ifeq (Darwin,$(UNAME))
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(call soname,$@ $(VERSION_MAJOR)) $(NOBLAS_HST) $(LDFLAGS) $(CLDFLAGS)
else
	$(LD) -o $@.$(VERSION_MAJOR).$(VERSION_MINOR) -shared $(EXTLDFLAGS) $(call soname,$@ $(VERSION_MAJOR)) $(NOBLAS_HST) $(LDFLAGS) $(CLDFLAGS)
endif
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@.$(VERSION_MAJOR)
	ln -fs $(notdir $@).$(VERSION_MAJOR).$(VERSION_MINOR) $@
else # static
$(OUTDIR)/libxsmmnoblas.$(LIBEXT): $(OUTDIR)/.make $(NOBLAS_HST)
	$(AR) -rs $@ $(NOBLAS_HST)
endif

.PHONY: samples
samples: cp2k nek smm wrap

.PHONY: cp2k
cp2k: lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: cp2k_mic
cp2k_mic: lib_mic
	@cd $(SPLDIR)/cp2k && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) KNC=1 TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: wrap
wrap: lib_hst
	@cd $(SPLDIR)/wrap && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=0 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: wrap_mic
wrap_mic: lib_mic
	@cd $(SPLDIR)/wrap && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) KNC=1 TRACE=0 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: nek
nek: lib_hst
	@cd $(SPLDIR)/nek && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: nek_mic
nek_mic: lib_mic
	@cd $(SPLDIR)/nek && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) KNC=1 TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: smm
smm: lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: smm_mic
smm_mic: lib_mic
	@cd $(SPLDIR)/smm && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) KNC=1 TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

# added for specfem sample
# will need option: make MNK="5 25" ..
.PHONY: specfem
specfem: lib_hst
	@cd $(SPLDIR)/specfem && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: specfem_mic
specfem_mic: lib_mic
	@cd $(SPLDIR)/specfem && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) KNC=1 TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: drytest
drytest: $(SPLDIR)/cp2k/cp2k-perf.sh $(SPLDIR)/smm/smmf-perf.sh \
	$(SPLDIR)/nek/axhm-perf.sh $(SPLDIR)/nek/grad-perf.sh $(SPLDIR)/nek/rstr-perf.sh

$(SPLDIR)/cp2k/cp2k-perf.sh: $(SPLDIR)/cp2k/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=cp2k-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23 4_6_9 13_5_7 24_3_36\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  SIZE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "else" >> $@
	@echo "  SIZE=0" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/cp2k.sh \$${MVALUE} \$${SIZE} 0 \$${NVALUE} \$${KVALUE} >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/smm/smmf-perf.sh: $(SPLDIR)/smm/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/smmf-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"23_23_23 4_6_9 13_5_7 24_3_36\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/smm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/axhm-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/axhm-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_6_9 8_8_8 13_13_13 16_8_13\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/axhm.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/grad-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/grad-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_6_9 8_8_8 13_13_13 16_8_13\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NMAX=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "for RUN in \$${RUNS} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/grad.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

$(SPLDIR)/nek/rstr-perf.sh: $(SPLDIR)/nek/.make $(ROOTDIR)/Makefile
	@echo "#!/bin/sh" > $@
	@echo >> $@
	@echo "HERE=\$$(cd \$$(dirname \$$0); pwd -P)" >> $@
	@echo "ECHO=\$$(which echo)" >> $@
	@echo "FILE=\$${HERE}/rstr-perf.txt" >> $@
ifneq (,$(strip $(INDICES)))
	@echo "RUNS=\"$(INDICES)\"" >> $@
	@echo "RUNT=\"$(INDICES)\"" >> $@
else
	@echo "RUNS=\"4_4_4 8_8_8\"" >> $@
	@echo "RUNT=\"7_7_7 10_10_10\"" >> $@
endif
	@echo >> $@
	@echo "if [ \"\" != \"\$$1\" ]; then" >> $@
	@echo "  FILE=\$$1" >> $@
	@echo "  shift" >> $@
	@echo "fi" >> $@
	@echo "cat /dev/null > \$${FILE}" >> $@
	@echo >> $@
	@echo "NRUN=1" >> $@
	@echo "NRUNS=\$$(\$${ECHO} \$${RUNS} | wc -w)" >> $@
	@echo "NRUNT=\$$(\$${ECHO} \$${RUNT} | wc -w)" >> $@
	@echo "NMAX=\$$((NRUNS*NRUNT))" >> $@
	@echo "for RUN1 in \$${RUNS} ; do" >> $@
	@echo "  for RUN2 in \$${RUNT} ; do" >> $@
	@echo "  MVALUE=\$$(\$${ECHO} \$${RUN1} | cut -d_ -f1)" >> $@
	@echo "  NVALUE=\$$(\$${ECHO} \$${RUN1} | cut -d_ -f2)" >> $@
	@echo "  KVALUE=\$$(\$${ECHO} \$${RUN1} | cut -d_ -f3)" >> $@
	@echo "  MMVALUE=\$$(\$${ECHO} \$${RUN2} | cut -d_ -f1)" >> $@
	@echo "  NNVALUE=\$$(\$${ECHO} \$${RUN2} | cut -d_ -f2)" >> $@
	@echo "  KKVALUE=\$$(\$${ECHO} \$${RUN2} | cut -d_ -f3)" >> $@
	@echo "  >&2 \$\$${ECHO} -n \"\$${NRUN} of \$${NMAX} (M=\$${MVALUE} N=\$${NVALUE} K=\$${KVALUE})... \"" >> $@
	@echo "  ERROR=\$$({ CHECK=1 \$${HERE}/rstr.sh \$${MVALUE} \$${NVALUE} \$${KVALUE} \$${MMVALUE} \$${NNVALUE} \$${KKVALUE} \$$* >> \$${FILE}; } 2>&1)" >> $@
	@echo "  RESULT=\$$?" >> $@
	@echo "  if [ 0 != \$${RESULT} ]; then" >> $@
	@echo "    \$${ECHO} \"FAILED(\$${RESULT}) \$${ERROR}\"" >> $@
	@echo "    exit 1" >> $@
	@echo "  else" >> $@
	@echo "    \$${ECHO} \"OK \$${ERROR}\"" >> $@
	@echo "  fi" >> $@
	@echo "  \$${ECHO} >> \$${FILE}" >> $@
	@echo "  NRUN=\$$((NRUN+1))" >> $@
	@echo "done" >> $@
	@echo "done" >> $@
	@echo >> $@
	@chmod +x $@

.PHONY: test
test: test-cp2k

.PHONY: perf
perf: perf-cp2k

.PHONY: test-all
test-all: tests test-cp2k test-smm test-nek test-wrap

.PHONY: build-tests
build-tests: lib_hst
	@cd $(TSTDIR) && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS)

.PHONY: tests
tests: build-tests
	@cd $(TSTDIR) && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) test

.PHONY: test-cpp
test-cpp: $(INCDIR)/libxsmm_source.h
	@cd $(SPLDIR)/cp2k && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=0 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) \
		ECXXFLAGS="-DUSE_HEADER_ONLY $(ECXXFLAGS)" clean compile

.PHONY: test-cp2k
test-cp2k: $(SPLDIR)/cp2k/cp2k-test.txt
$(SPLDIR)/cp2k/cp2k-test.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	$(info ========================)
	$(info Running CP2K Code Sample)
	$(info ========================)
	@cd $(SPLDIR)/cp2k && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) cp2k
	@$(SPLDIR)/cp2k/cp2k-perf.sh $@ $(shell echo $$(($(TESTSIZE) * 128)))

.PHONY: perf-cp2k
perf-cp2k: $(SPLDIR)/cp2k/cp2k-perf.txt
$(SPLDIR)/cp2k/cp2k-perf.txt: $(SPLDIR)/cp2k/cp2k-perf.sh lib_hst
	@cd $(SPLDIR)/cp2k && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) cp2k
	@$(SPLDIR)/cp2k/cp2k-perf.sh $@

.PHONY: test-wrap
test-wrap: wrap
	@cd $(SPLDIR)/wrap && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=0 \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) test

.PHONY: test-smm
ifneq (,$(strip $(FC)))
test-smm: $(SPLDIR)/smm/smm-test.txt
$(SPLDIR)/smm/smm-test.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	$(info =======================)
	$(info Running SMM Code Sample)
	$(info =======================)
	@cd $(SPLDIR)/smm && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) smm
	@$(SPLDIR)/smm/smmf-perf.sh $@ $(shell echo $$(($(TESTSIZE) * -128)))
endif

.PHONY: perf-smm
ifneq (,$(strip $(FC)))
perf-smm: $(SPLDIR)/smm/smmf-perf.txt
$(SPLDIR)/smm/smmf-perf.txt: $(SPLDIR)/smm/smmf-perf.sh lib_hst
	@cd $(SPLDIR)/smm && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) smm
	@$(SPLDIR)/smm/smmf-perf.sh $@
endif

.PHONY: test-nek
ifneq (,$(strip $(FC)))
test-nek: $(SPLDIR)/nek/axhm-perf.txt $(SPLDIR)/nek/grad-perf.txt $(SPLDIR)/nek/rstr-perf.txt
$(SPLDIR)/nek/axhm-perf.txt: $(SPLDIR)/nek/axhm-perf.sh lib_hst
	$(info =======================)
	$(info Running NEK/AXHM Sample)
	$(info =======================)
	@cd $(SPLDIR)/nek && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) axhm
	@$(SPLDIR)/nek/axhm-perf.sh $@ $(shell echo $$(($(TESTSIZE) * -128)))
$(SPLDIR)/nek/grad-perf.txt: $(SPLDIR)/nek/grad-perf.sh lib_hst
	$(info =======================)
	$(info Running NEK/GRAD Sample)
	$(info =======================)
	@cd $(SPLDIR)/nek && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) grad
	@$(SPLDIR)/nek/grad-perf.sh $@ $(shell echo $$(($(TESTSIZE) * -128)))
$(SPLDIR)/nek/rstr-perf.txt: $(SPLDIR)/nek/rstr-perf.sh lib_hst
	$(info =======================)
	$(info Running NEK/RSTR Sample)
	$(info =======================)
	@cd $(SPLDIR)/nek && $(MAKE) --no-print-directory COMPATIBLE=$(COMPATIBLE) THREADS=$(THREADS) \
		DEPSTATIC=$(STATIC) SYM=$(SYM) DBG=$(DBG) IPO=$(IPO) SSE=$(SSE) AVX=$(AVX) MIC=$(MIC) OFFLOAD=$(OFFLOAD) TRACE=$(TRACE) \
		EFLAGS=$(EFLAGS) ELDFLAGS=$(ELDFLAGS) ECXXFLAGS=$(ECXXFLAGS) ECFLAGS=$(ECFLAGS) EFCFLAGS=$(EFCFLAGS) rstr
	@$(SPLDIR)/nek/rstr-perf.sh $@ $(shell echo $$(($(TESTSIZE) * -128)))
endif

$(DOCDIR)/libxsmm.pdf: $(DOCDIR)/.make $(ROOTDIR)/README.md
	$(eval TMPFILE = $(shell mktemp fileXXXXXX))
	@mv $(TMPFILE) $(TMPFILE).tex
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' > \
		$(TMPFILE).tex
	@iconv -t utf-8 $(ROOTDIR)/README.md \
	| sed \
		-e 's/\[\[..*\](..*)\]//g' \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--latex-engine=xelatex --template=$(TMPFILE).tex --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="LIBXSMM Documentation" \
		-V author-meta="Hans Pabst, Alexander Heinecke" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE).tex

$(DOCDIR)/cp2k.pdf: $(DOCDIR)/.make $(ROOTDIR)/documentation/cp2k.md
	$(eval TMPFILE = $(shell mktemp fileXXXXXX))
	@mv $(TMPFILE) $(TMPFILE).tex
	@pandoc -D latex \
	| sed \
		-e 's/\(\\documentclass\[..*\]{..*}\)/\1\n\\pagenumbering{gobble}\n\\RedeclareSectionCommands[beforeskip=-1pt,afterskip=1pt]{subsection,subsubsection}/' \
		-e 's/\\usepackage{listings}/\\usepackage{listings}\\lstset{basicstyle=\\footnotesize\\ttfamily}/' > \
		$(TMPFILE).tex
	@iconv -t utf-8 $(ROOTDIR)/documentation/cp2k.md \
	| sed \
		-e 's/\[\[..*\](..*)\]//g' \
		-e 's/\[!\[..*\](..*)\](..*)//g' \
		-e 's/<sub>/~/g' -e 's/<\/sub>/~/g' \
		-e 's/<sup>/^/g' -e 's/<\/sup>/^/g' \
		-e 's/----*//g' \
	| pandoc \
		--latex-engine=xelatex --template=$(TMPFILE).tex --listings \
		-f markdown_github+implicit_figures+all_symbols_escapable+subscript+superscript \
		-V documentclass=scrartcl \
		-V title-meta="CP2K with LIBXSMM" \
		-V author-meta="Hans Pabst" \
		-V classoption=DIV=45 \
		-V linkcolor=black \
		-V citecolor=black \
		-V urlcolor=black \
		-o $@
	@rm $(TMPFILE).tex

.PHONY: documentation
documentation: $(DOCDIR)/libxsmm.pdf $(DOCDIR)/cp2k.pdf

.PHONY: clean-minimal
clean-minimal:
	@rm -rf $(SCRDIR)/__pycache__
	@rm -f $(SCRDIR)/libxsmm_utilities.pyc
	@touch $(INCDIR)/.make 2> /dev/null || true
	@touch $(SPLDIR)/cp2k/.make
	@touch $(SPLDIR)/smm/.make
	@touch $(SPLDIR)/nek/.make

.PHONY: clean
clean: clean-minimal
	@rm -f $(OBJECTS) $(FTNOBJS) $(SRCFILES_KERNELS)
	@rm -f $(BLDDIR)/libxsmm_dispatch.h
	@if [ "" = "$$(find build -type f -not -name .make 2> /dev/null)" ]; then \
		rm -rf $(BLDDIR); \
	fi

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
endif
endif
ifneq ($(abspath $(BINDIR)),$(ROOTDIR))
ifneq ($(abspath $(BINDIR)),$(abspath .))
	@rm -rf $(BINDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR)))
	@rm -f $(OUTDIR)/libxsmm.$(LIBEXT)* $(OUTDIR)/mic/libxsmm.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmf.$(LIBEXT)* $(OUTDIR)/mic/libxsmmf.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmext.$(LIBEXT)* $(OUTDIR)/mic/libxsmmext.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmnoblas.$(LIBEXT)* $(OUTDIR)/mic/libxsmmnoblas.$(LIBEXT)*
	@rm -f $(OUTDIR)/libxsmmgen.$(LIBEXT)*
endif
ifneq (,$(wildcard $(BINDIR)))
	@rm -f $(BINDIR)/libxsmm_*_generator
endif
	@rm -f *.gcno *.gcda *.gcov
	@rm -f $(SPLDIR)/cp2k/cp2k-perf.sh
	@rm -f $(SPLDIR)/smm/smmf-perf.sh
	@rm -f $(SPLDIR)/nek/grad-perf.sh
	@rm -f $(SPLDIR)/nek/axhm-perf.sh
	@rm -f $(SPLDIR)/nek/rstr-perf.sh
	@rm -f $(INCDIR)/libxsmm_config.h
	@rm -f $(INCDIR)/libxsmm_source.h
	@rm -f $(INCDIR)/libxsmm.modmic
	@rm -f $(INCDIR)/libxsmm.mod
	@rm -f $(INCDIR)/libxsmm.f
	@rm -f $(INCDIR)/libxsmm.h
	@rm -f $(INCDIR)/.make
	@rm -f $(DOCDIR)/.make
	@rm -f .make .state

.PHONY: clean-all
clean-all: clean
	@cd $(TSTDIR)           && $(MAKE) --no-print-directory clean-minimal
	@cd $(SPLDIR)/cp2k      && $(MAKE) --no-print-directory clean-minimal
	@cd $(SPLDIR)/dispatch  && $(MAKE) --no-print-directory clean-minimal
	@cd $(SPLDIR)/nek       && $(MAKE) --no-print-directory clean-minimal
	@cd $(SPLDIR)/smm       && $(MAKE) --no-print-directory clean-minimal
	@cd $(SPLDIR)/wrap      && $(MAKE) --no-print-directory clean-minimal

.PHONY: realclean-all
realclean-all: realclean
	@cd $(TSTDIR)           && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/barrier   && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/cp2k      && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/dispatch  && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/dnn       && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/nek       && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/smm       && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/specfem   && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/transpose && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/wrap      && $(MAKE) --no-print-directory realclean
	@cd $(SPLDIR)/xgemm     && $(MAKE) --no-print-directory realclean

# Dummy prefix
ifneq (,$(strip $(PREFIX)))
INSTALL_ROOT = $(PREFIX)
else
INSTALL_ROOT = .
endif

.PHONY: install-minimal
install-minimal: libxsmm
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXSMM installing binaries..."
	@mkdir -p $(INSTALL_ROOT)/$(POUTDIR) $(INSTALL_ROOT)/$(PBINDIR) $(INSTALL_ROOT)/$(PINCDIR)
	@cp -v $(OUTDIR)/libxsmmnoblas.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmmnoblas.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmmgen.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmmgen.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmmext.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmmext.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmmf.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmmf.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmm.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@cp -v $(OUTDIR)/libxsmm.$(SLIBEXT)  $(INSTALL_ROOT)/$(POUTDIR) 2> /dev/null || true
	@if [ -e $(OUTDIR)/mic/libxsmmnoblas.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmmnoblas.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmnoblas.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmmnoblas.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmext.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmmext.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmext.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmmext.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmf.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmmf.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmmf.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmmf.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmm.$(DLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmm.$(DLIBEXT)* $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@if [ -e $(OUTDIR)/mic/libxsmm.$(SLIBEXT) ]; then \
		mkdir -p $(INSTALL_ROOT)/$(POUTDIR)/mic; \
		cp -v $(OUTDIR)/mic/libxsmm.$(SLIBEXT) $(INSTALL_ROOT)/$(POUTDIR)/mic; \
	fi
	@echo
	@echo "LIBXSMM installing interface..."
	@cp -v $(BINDIR)/libxsmm_*_generator $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(INCDIR)/*.mod* $(INSTALL_ROOT)/$(PINCDIR) 2> /dev/null || true
	@cp -v $(INCDIR)/libxsmm*.h $(INSTALL_ROOT)/$(PINCDIR)
	@cp -v $(INCDIR)/libxsmm.f $(INSTALL_ROOT)/$(PINCDIR)
	@echo
	@echo "LIBXSMM installing stand-alone generators..."
	@cp -v $(BINDIR)/libxsmm_*_generator $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
endif

.PHONY: install
install: install-minimal
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXSMM installing documentation..."
	@mkdir -p $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/$(DOCDIR)/*.pdf $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/$(DOCDIR)/*.md $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/version.txt $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/README.md $(INSTALL_ROOT)/$(PDOCDIR)
	@cp -v $(ROOTDIR)/LICENSE $(INSTALL_ROOT)/$(PDOCDIR)
endif

.PHONY: install-all
install-all: install samples
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXSMM installing samples..."
	@cp -v $(addprefix $(SPLDIR)/cp2k/,cp2k cp2k.sh cp2k-perf* cp2k-plot.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/wrap/,dgemm-blas dgemm-blas.sh dgemm-wrap dgemm-wrap.sh dgemm-test.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/dispatch/,dispatch dispatch.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/nek/,axhm grad rstr *.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,smm smm.sh smm-perf* smmf-perf.sh smm-plot.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,specialized specialized.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,dispatched dispatched.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,inlined inlined.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
	@cp -v $(addprefix $(SPLDIR)/smm/,blas blas.sh) $(INSTALL_ROOT)/$(PBINDIR) 2> /dev/null || true
endif

.PHONY: install-dev
install-dev: install-all build-tests
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXSMM installing tests..."
	@mkdir -p $(INSTALL_ROOT)/$(PTSTDIR)
	@cp -v $(basename $(shell ls -1 ${TSTDIR}/*.c 2> /dev/null | tr "\n" " ")) $(INSTALL_ROOT)/$(PTSTDIR) 2> /dev/null || true
endif

.PHONY: install-artifacts
install-artifacts: install-dev
ifneq ($(abspath $(INSTALL_ROOT)),$(abspath .))
	@echo
	@echo "LIBXSMM installing artifacts..."
	@mkdir -p $(INSTALL_ROOT)/$(PDOCDIR)/artifacts
	@cp -v .state $(INSTALL_ROOT)/$(PDOCDIR)/artifacts/make.txt
endif

