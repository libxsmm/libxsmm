# Export all variables to sub-make processes.
#.EXPORT_ALL_VARIABLES: #export

# Automatically disable parallel builds
# depending on the version of GNU Make.
# MAKE_PARALLEL=0: disable explicitly
# MAKE_PARALLEL=1: enable explicitly
ifeq (0,$(MAKE_PARALLEL))
.NOTPARALLEL:
else ifeq (,$(MAKE_PARALLEL))
ifneq (3.82,$(firstword $(sort $(MAKE_VERSION) 3.82)))
.NOTPARALLEL:
endif
endif

ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ../..
SRCDIR = $(ROOTDIR)
INCDIR = .
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(DEPDIR)/include

# Fortran code here does not allow for PEDANTIC=2
override PEDANTIC = 1
BLAS = 1
OMP = 1
SYM = 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh)
CPPSRCS := $(wildcard $(SRCDIR)/*.cpp)
CXXSRCS := $(wildcard $(SRCDIR)/*.cxx)
CCXSRCS := $(wildcard $(SRCDIR)/*.cx)
CSOURCS := $(wildcard $(SRCDIR)/*.c)
FXXSRCS := $(wildcard $(SRCDIR)/*.f)
F77SRCS := $(wildcard $(SRCDIR)/*.F)
F90SRCS := $(wildcard $(SRCDIR)/*.f90) $(wildcard $(SRCDIR)/*.F90)
CPPOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CPPSRCS:.cpp=-cpp.o)))
CXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CXXSRCS:.cxx=-cxx.o)))
CCXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CCXSRCS:.cc=-cc.o)))
COBJCTS := $(patsubst %,$(BLDDIR)/%,$(notdir $(CSOURCS:.c=-c.o)))
FXXOBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(FXXSRCS:.f=-f.o)))
F77OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F77SRCS:.F=-f77.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.f90=-f90.o)))
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90OBJS:.F90=-f90.o)))
SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
FTNSRCS := $(FXXSRCS) $(F77SRCS) $(F90SRCS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS))) $(addsuffix .modmic,$(basename $(FTNSRCS)))
FTNOBJS := $(FXXOBJS) $(F77OBJS) $(F90OBJS)

ifneq (,$(strip $(FC)))
  XFILES := $(OUTDIR)/axhm $(OUTDIR)/grad $(OUTDIR)/rstr $(OUTDIR)/libstream_update_kernels.$(SLIBEXT)
endif

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

$(OUTDIR)/libstream_update_kernels.$(SLIBEXT): $(OUTDIR)/.make $(BLDDIR)/stream_update_kernels-c.o $(BLDDIR)/stream_update_kernels-f.o
	$(AR) -rs $@ $(BLDDIR)/stream_update_kernels-c.o $(BLDDIR)/stream_update_kernels-f.o

$(OUTDIR)/axhm: $(OUTDIR)/.make $(OUTDIR)/libstream_update_kernels.$(SLIBEXT) $(BLDDIR)/axhm-f.o $(BLDDIR)/mxm_std-f.o $(LIBNAME)f.$(LIBEXT) $(LIBNAME).$(LIBEXT)
	$(FLD) -o $@ $(BLDDIR)/axhm-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.$(SLIBEXT) \
		$(call abslib,$(LIBNAME)f.$(LIBEXT)) $(call abslib,$(LIBNAME).$(LIBEXT)) \
		$(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)

$(OUTDIR)/grad: $(OUTDIR)/.make $(OUTDIR)/libstream_update_kernels.$(SLIBEXT) $(BLDDIR)/grad-f.o $(BLDDIR)/mxm_std-f.o $(LIBNAME)f.$(LIBEXT) $(LIBNAME).$(LIBEXT)
	$(FLD) -o $@ $(BLDDIR)/grad-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.$(SLIBEXT) \
		$(call abslib,$(LIBNAME)f.$(LIBEXT)) $(call abslib,$(LIBNAME).$(LIBEXT)) \
		$(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)

$(OUTDIR)/rstr: $(OUTDIR)/.make $(OUTDIR)/libstream_update_kernels.$(SLIBEXT) $(BLDDIR)/rstr-f.o $(BLDDIR)/mxm_std-f.o $(LIBNAME)f.$(LIBEXT) $(LIBNAME).$(LIBEXT)
	$(FLD) -o $@ $(BLDDIR)/rstr-f.o $(BLDDIR)/mxm_std-f.o $(OUTDIR)/libstream_update_kernels.$(SLIBEXT) \
		$(call abslib,$(LIBNAME)f.$(LIBEXT)) $(call abslib,$(LIBNAME).$(LIBEXT)) \
		$(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)

$(BLDDIR)/mxm_std-f.o: $(SRCDIR)/mxm_std.f .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/stream_update_kernels-f.o: $(SRCDIR)/stream_update_kernels.f .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/axhm-f.o: $(SRCDIR)/axhm.f $(BLDDIR)/stream_update_kernels-f.o .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/grad-f.o: $(SRCDIR)/grad.f $(BLDDIR)/stream_update_kernels-f.o .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/rstr-f.o: $(SRCDIR)/rstr.f $(BLDDIR)/stream_update_kernels-f.o .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* fit.log *.dat
	@rm -f $(BLDDIR)/*.gcno $(BLDDIR)/*.gcda $(BLDDIR)/*.gcov
endif
	@rm -f .make .state

.PHONY: realclean
realclean: clean
ifneq ($(abspath $(OUTDIR)),$(ROOTDIR))
ifneq ($(abspath $(OUTDIR)),$(abspath .))
	@rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@rm -f $(OUTDIR)/libxsmm.$(DLIBEXT) $(OUTDIR)/*.stackdump
	@rm -f $(XFILES) $(MODULES)
endif

