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
INCDIR = $(DEPDIR)/include
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(BLDDIR) -I$(DEPDIR)/include

BLAS ?= 1
OMP ?= 1
SYM ?= 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

ifeq (0,$(MIC))
	LIBNAME = $(DEPDIR)/lib/libxsmm
else ifeq (0,$(OFFLOAD))
	LIBNAME = $(DEPDIR)/lib/mic/libxsmm
else
	LIBNAME = $(DEPDIR)/lib/libxsmm
endif

DEPSTATIC ?= $(STATIC)
ifneq (0,$(DEPSTATIC))
	ifneq (,$(strip $(wildcard $(LIBNAME).$(SLIBEXT))))
		LIBEXT = $(SLIBEXT)
	else # fallback
		LIBEXT = $(DLIBEXT)
	endif
else
	ifneq (,$(strip $(wildcard $(LIBNAME).$(DLIBEXT))))
		LIBEXT = $(DLIBEXT)
	else # fallback
		LIBEXT = $(SLIBEXT)
	endif
endif

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(shell ls -1 $(INCDIR)/*.h   2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(INCDIR)/*.hh  2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.h   2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hpp 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hxx 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.hh  2> /dev/null | tr "\n" " ")
CPPSRCS := $(shell ls -1 $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CXXSRCS := $(shell ls -1 $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CCXSRCS := $(shell ls -1 $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CSOURCS := $(shell ls -1 $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
FXXSRCS := $(shell ls -1 $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
F77SRCS := $(shell ls -1 $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F90SRCS := $(shell ls -1 $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ") \
           $(shell ls -1 $(SRCDIR)/*.F90 2> /dev/null | tr "\n" " ")
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

.PHONY: all
all: $(OUTDIR)/$(OUTNAME)

$(OUTDIR)/$(OUTNAME): $(OUTDIR)/.make $(OBJECTS)
	$(LD) -o $@ $(OBJECTS) $(abspath $(LIBNAME).$(LIBEXT)) $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(LIBNAME).$(LIBEXT)
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(TARGET) -c $< -o $@

.PHONY: clean
clean:
	@rm -f $(OBJECTS) $(FTNOBJS)
	@rm -f fit.log *.dat

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
else
	@rm -f $(OUTDIR)/$(OUTNAME)
endif
else
	@rm -f $(OUTDIR)/$(OUTNAME)
endif
	@rm -f $(MODULES) $(ROOTDIR)/cp2k-perf.sh
	@rm -f *.gcno *.gcda *.gcov
	@rm -f $(OUTDIR)/libxsmm.$(DLIBEXT)
	@rm -f $(OUTDIR)/*.bin
	@rm -f $(OUTDIR)/.make

install: all clean
	@cp $(DEPDIR)/lib/libxsmm.$(DLIBEXT) $(OUTDIR) 2> /dev/null || true
ifeq (0,$(OFFLOAD))
ifneq (0,$(MIC))
	@cp $(DEPDIR)/lib/mic/libxsmm.$(DLIBEXT) $(OUTDIR) 2> /dev/null || true
endif
endif
	@cp $(ROOTDIR)/$(OUTNAME).sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf-jit.plt $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.plt $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-perf.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-plot.sh $(OUTDIR) 2> /dev/null || true
	@cp $(ROOTDIR)/$(OUTNAME)-test.sh $(OUTDIR) 2> /dev/null || true

