ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../../..
SRCDIR = .
INCDIR = .
BLDDIR = obj
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)

OMP = 1
SYM = 1

BLAS = $(if $(filter-out 0,$(OMP)),2,1)
BLAS_STATIC = 0

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

ifneq (,$(strip $(wildcard $(LIBNAME).$(SLIBEXT))))
  EXTDEP := $(LIBNAME)ext.$(SLIBEXT)
  LIBDEP := $(LIBNAME).$(SLIBEXT)
endif

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)

ifneq (Darwin,$(UNAME))
  XWRAP ?= $(if $(filter-out 0,$(NOBLAS)),0,2)
else
  XWRAP ?= 0
  ifneq (0,$(XWRAP))
    $(info ==========================================================)
    $(info The static link-time wrapper is not supported under macOS!)
    $(info ==========================================================)
  endif
endif
ifneq (0,$(XWRAP))
ifneq (,$(PREC))
  XWRAP := $(PREC)
endif
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh) \
           $(DEPDIR)/include/libxsmm_source.h
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
CXXSRCS := $(wildcard $(SRCDIR)/*.cxx)
CCXSRCS := $(wildcard $(SRCDIR)/*.cc)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
CPPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
ifneq (,$(strip $(FC)))
FXXSRCS := $(wildcard $(SRCDIR)/*.f)
F77SRCS := $(wildcard $(SRCDIR)/*.F)
F90SRCS := $(wildcard $(SRCDIR)/*.f90) $(wildcard $(SRCDIR)/*.F90)
FXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FXXSRCS:.f=-f.o)))
F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90OBJS:.F90=-f90.o)))
endif
SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
FTNSRCS := $(FXXSRCS) $(F77SRCS) $(F90SRCS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS)))
FTNOBJS := $(FXXOBJS) $(F77OBJS) $(F90OBJS)
XFILES := \
  $(OUTDIR)/dgemm_batch_strided-blas $(OUTDIR)/dgemm_batch_strided-wrap \
  $(OUTDIR)/dgemm_batch-blas $(OUTDIR)/dgemm_batch-wrap \
  $(OUTDIR)/dgemm-blas $(OUTDIR)/dgemm-wrap \
  $(OUTDIR)/dgemv-blas $(OUTDIR)/dgemv-wrap

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

ifneq (0,$(XWRAP))
  MKL_VERSION_NUM := $(call version,$(call prepget,$(SRCDIR)/dgemm_batch_strided.c,__INTEL_MKL__,__INTEL_MKL_MINOR__,__INTEL_MKL_UPDATE__))
  ifneq (2,$(XWRAP))
    ifneq (0,$(shell echo "$$((20200002<=$(MKL_VERSION_NUM)))"))
      WRAP_GEMM_BATCH_STRIDED := -Wl,--wrap=dgemm_batch_strided_,--wrap=sgemm_batch_strided_
    endif
    ifneq (0,$(shell echo "$$((110300<=$(MKL_VERSION_NUM)))"))
      WRAP_GEMM_BATCH := -Wl,--wrap=dgemm_batch_,--wrap=sgemm_batch_
    endif
    WRAP_GEMM := -Wl,--wrap=dgemm_,--wrap=sgemm_
    WRAP_GEMV := -Wl,--wrap=dgemv_,--wrap=sgemv_
  else
    ifneq (0,$(shell echo "$$((20200002<=$(MKL_VERSION_NUM)))"))
      WRAP_GEMM_BATCH_STRIDED := -Wl,--wrap=dgemm_batch_strided_
    endif
    ifneq (0,$(shell echo "$$((110300<=$(MKL_VERSION_NUM)))"))
      WRAP_GEMM_BATCH := -Wl,--wrap=dgemm_batch_
    endif
    WRAP_GEMM := -Wl,--wrap=dgemm_
    WRAP_GEMV := -Wl,--wrap=dgemv_
  endif
  DFLAGS += -DWRAP
endif

ifeq (0,$(NOBLAS))
$(OUTDIR)/dgemm_batch_strided-blas: $(BLDDIR)/dgemm_batch_strided-c.o $(OUTDIR)/.make
	$(LD) -o $@ $< $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
else
.PHONY: $(OUTDIR)/dgemm_batch_strided-blas
endif

ifeq (0,$(NOBLAS))
$(OUTDIR)/dgemm_batch-blas: $(BLDDIR)/dgemm_batch-c.o $(OUTDIR)/.make
	$(LD) -o $@ $< $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
else
.PHONY: $(OUTDIR)/dgemm_batch-blas
endif

ifeq (0,$(NOBLAS))
$(OUTDIR)/dgemm-blas: $(BLDDIR)/dgemm-c.o $(OUTDIR)/.make
	$(LD) -o $@ $< $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
else
.PHONY: $(OUTDIR)/dgemm-blas
endif

ifeq (0,$(NOBLAS))
$(OUTDIR)/dgemv-blas: $(BLDDIR)/dgemv-c.o $(OUTDIR)/.make
	$(LD) -o $@ $< $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
else
.PHONY: $(OUTDIR)/dgemv-blas
endif

ifneq (,$(WRAP_GEMM_BATCH_STRIDED))
$(OUTDIR)/dgemm_batch_strided-wrap: $(BLDDIR)/dgemm_batch_strided-c.o $(OUTDIR)/.make $(EXTDEP)
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM_BATCH_STRIDED)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM_BATCH_STRIDED) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM_BATCH_STRIDED)
endif
else
.PHONY: $(OUTDIR)/dgemm_batch_strided-wrap
endif

ifneq (,$(WRAP_GEMM_BATCH))
$(OUTDIR)/dgemm_batch-wrap: $(BLDDIR)/dgemm_batch-c.o $(OUTDIR)/.make $(EXTDEP)
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM_BATCH)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM_BATCH) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM_BATCH)
endif
else
.PHONY: $(OUTDIR)/dgemm_batch-wrap
endif

ifneq (,$(WRAP_GEMM))
$(OUTDIR)/dgemm-wrap: $(BLDDIR)/dgemm-c.o $(OUTDIR)/.make $(EXTDEP)
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMM)
endif
else
.PHONY: $(OUTDIR)/dgemm-wrap
endif

ifneq (,$(WRAP_GEMV))
$(OUTDIR)/dgemv-wrap: $(BLDDIR)/dgemv-c.o $(OUTDIR)/.make $(EXTDEP)
ifneq (0,$(OMP))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMV)
else ifneq (,$(strip $(OMPLIB)))
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMV) \
		$(XLIB_BEGIN) $(OMPLIB) $(XLIB_END)
else # should not happen
	$(LD) -o $@ $< $(EXTLIB) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(WRAP_GEMV)
endif
else
.PHONY: $(OUTDIR)/dgemv-wrap
endif

.PHONY: test
test: $(ROOTDIR)/wrap-test.sh $(XFILES)
	@$(ROOTDIR)/wrap-test.sh dgemm_batch_strided                                             $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm_batch_strided 35 16 20 35 35 16    0    0   0 1024  1 0.0 $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm_batch_strided 20 20 32 24 32 24 1000 1000 500 2000  1 0.0 $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm_batch_strided 24 23 21 32 32 32    0    0   0  999 -1 0.5 $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm_batch                               $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm_batch 35 16 20 35 35 16 1024  1 0.0 $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm_batch 20 20 32 24 32 24 2000  1 0.0 $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm_batch 24 23 21 32 32 32  999 -1 0.5 $(shell echo $$(($(TESTSIZE) * 1)))
	@$(ROOTDIR)/wrap-test.sh dgemm                                $(shell echo $$(($(TESTSIZE) * 10)))
	@$(ROOTDIR)/wrap-test.sh dgemm 350  16  20 350  35 350  1 0.0 $(shell echo $$(($(TESTSIZE) * 10)))
	@$(ROOTDIR)/wrap-test.sh dgemm 200 200 200 256 256 256  1 0.0 $(shell echo $$(($(TESTSIZE) * 10)))
	@$(ROOTDIR)/wrap-test.sh dgemm  24  23  21  32  32  32 -1 0.5 $(shell echo $$(($(TESTSIZE) * 10)))
	@$(ROOTDIR)/wrap-test.sh dgemv                     $(shell echo $$(($(TESTSIZE) * 1000)))
	@$(ROOTDIR)/wrap-test.sh dgemv 350  20 350 1 1 1 0 $(shell echo $$(($(TESTSIZE) * 1000)))
	@$(ROOTDIR)/wrap-test.sh dgemv 200 200 256 1 1 1 0 $(shell echo $$(($(TESTSIZE) * 1000)))
	@$(ROOTDIR)/wrap-test.sh dgemv  24  21  32 2 2 1 1 $(shell echo $$(($(TESTSIZE) * 1000)))

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
	@-rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@-rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* *.dat *.log
	@-rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif

.PHONY: realclean
realclean: clean
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@-rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@-rm -f $(OUTDIR)/libxsmm.$(DLIBEXT) $(OUTDIR)/*.stackdump
	@-rm -f $(XFILES) $(MODULES)
endif

.PHONY: deepclean
deepclean: realclean
	@-rm -f .make .state
