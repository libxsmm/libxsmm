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
F90OBJS := $(patsubst %,$(BLDDIR)/%,$(notdir $(F90SRCS:.F90=-f90.o)))

SOURCES := $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)
OBJECTS := $(CPPOBJS) $(CXXOBJS) $(CCXOBJS) $(COBJCTS)
FTNSRCS := $(FXXSRCS) $(F77SRCS) $(F90SRCS)
MODULES := $(addsuffix .mod,$(basename $(FTNSRCS))) $(addsuffix .modmic,$(basename $(FTNSRCS)))
FTNOBJS := $(FXXOBJS) $(F77OBJS) $(F90OBJS)
XFILES := $(OUTDIR)/$(OUTNAME)

# specfem example specific
MODULES += my_libxsmm.mod constants.mod specfem_par.mod my_libxsmm.modmic constants.modmic specfem_par.modmic
DFLAGS += -DFORCE_VECTORIZATION

# OpenMP directives support
ifneq (0,$(OMP))
DFLAGS += -DUSE_OPENMP
endif

# fixes library paths: substitutes path name from ../mkl/lib/intel64/.. to ../mkl/lib/mic/..
ifneq (0,$(KNC))
ifneq (0,$(MPSS))
lib_intel := mkl/lib/intel64
lib_mic := mkl/lib/mic
LDFLAGS_MIC = $(subst $(lib_intel),$(lib_mic),$(LDFLAGS))
endif
endif

##
## targets
##

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

$(OUTDIR)/specfem: $(OUTDIR)/.make $(FTNOBJS) $(LIBNAME).$(LIBEXT)
ifneq (0,$(KNC))
ifneq (0,$(MPSS))
	@echo ""
	@echo "building MIC/KNC version"
	@echo ""
	$(FLD) -o $@ -mmic $(FTNOBJS) $(call abslib,$(LIBNAME)f.$(LIBEXT)) $(call abslib,$(LIBNAME).$(LIBEXT)) $(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS_MIC) $(FLDFLAGS) $(ELDFLAGS)
endif
endif
ifeq (0,$(KNC))
ifeq (0,$(MPSS))
	@echo ""
	@echo "building host version"
	@echo ""
	$(FLD) -o $@ $(FTNOBJS) $(call abslib,$(LIBNAME)f.$(LIBEXT)) $(call abslib,$(LIBNAME).$(LIBEXT)) $(FCMTFLAGS) $(SLDFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
endif
endif

.PHONY: clean
clean:
ifneq ($(abspath $(BLDDIR)),$(ROOTDIR))
ifneq ($(abspath $(BLDDIR)),$(abspath .))
	@rm -rf $(BLDDIR)
endif
endif
ifneq (,$(wildcard $(BLDDIR))) # still exists
	@rm -f $(OBJECTS) $(OBJECTX) $(FTNOBJS) $(FTNOBJX) *__genmod.* fit.log *.dat *.mhd
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



##
## dependencies
##
$(BLDDIR)/compute_forces_Dev-f90.o: $(BLDDIR)/specfem-f90.o
$(BLDDIR)/compute_forces_noDev-f90.o: $(BLDDIR)/specfem-f90.o
$(BLDDIR)/compute_forces_xsmm_dispatch-f90.o: $(BLDDIR)/specfem-f90.o
$(BLDDIR)/compute_forces_xsmm_prefetch-f90.o: $(BLDDIR)/specfem-f90.o
$(BLDDIR)/compute_forces_xsmm_static-f90.o: $(BLDDIR)/specfem-f90.o


##
## rules
##
ifneq (0,$(KNC))
ifneq (0,$(MPSS))
$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) -mmic -c $< -o $@
endif
endif
ifeq (0,$(KNC))
ifeq (0,$(MPSS))
$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@
endif
endif
