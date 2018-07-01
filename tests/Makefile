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
DEPDIR = $(ROOTDIR)/..
SRCDIR = $(ROOTDIR)
INCDIR = $(ROOTDIR)
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -DLIBXSMM_GEMM_CONST
IFLAGS = -I$(INCDIR) -I$(DEPDIR)/include -I$(DEPDIR)/src

BLAS ?= 1
OMP ?= 1
SYM ?= 1
MIX ?= 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh)
CPPSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cpp 2>/dev/null | tr "\n" " ")
CPPSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cpp 2>/dev/null | tr "\n" " ")
CXXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cxx 2>/dev/null | tr "\n" " ")
CXXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cxx 2>/dev/null | tr "\n" " ")
CCXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cc  2>/dev/null | tr "\n" " ")
CCXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cc  2>/dev/null | tr "\n" " ")
CSOURCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.c   2>/dev/null | tr "\n" " ")
CSOURCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.c   2>/dev/null | tr "\n" " ")
FXXSRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f   2>/dev/null | tr "\n" " ")
FXXSRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f   2>/dev/null | tr "\n" " ")
F77SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.F   2>/dev/null | tr "\n" " ")
F77SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.F   2>/dev/null | tr "\n" " ")
F90SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f90 2>/dev/null | tr "\n" " ") \
           $(shell grep -L $(FMAIN) $(SRCDIR)/*.F90 2>/dev/null | tr "\n" " ")
F90SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f90 2>/dev/null | tr "\n" " ") \
           $(shell grep -l $(FMAIN) $(SRCDIR)/*.F90 2>/dev/null | tr "\n" " ")
MODULES := $(addsuffix    .mod,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS))) \
           $(addsuffix .modmic,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS)))
OBJECTS := $(call objname, $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS))
OBJECTX := $(call objname, $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX))
FTNOBJS := $(call objname, $(FXXSRCS) $(F77SRCS) $(F90SRCS))
FTNOBJX := $(call objname, $(FXXSRCX) $(F77SRCX) $(F90SRCX))
XFILES  := $(addprefix $(OUTDIR)/,$(basename $(notdir \
           $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX) \
           $(FXXSRCX) $(F77SRCX) $(F90SRCX))))

.PHONY: all
all: $(XFILES)

.PHONY: compile
compile: $(OBJECTS) $(FTNOBJS)

.PHONY: tests
tests: test

.PHONY: test
test: $(OUTDIR)/.make $(OUTDIR)/test.sh $(XFILES)
	@$(OUTDIR)/test.sh $(TEST)

# determine header-only tests (to avoid linking against LIBXSMM libraries; see below)
HEADER_ONLY = $(basename $(notdir $(shell grep -H libxsmm_source *.c | cut -d: -f1)))

define DEFINE_LINK_LD_RULE
ifneq (,$(wildcard $(LIBDEP)))
ifneq (,$(wildcard $(EXTDEP)))
$(1): $(2) $(call objname, $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)) $(dir $(1))/.make \
      $(if $(filter $(1),$(HEADER_ONLY)),$(NULL),$(LIBDEP) $(EXTDEP))
	$(LD) $(SLDFLAGS) -o $(1) $(2) \
		$(if $(filter $(1),$(HEADER_ONLY)),$(NULL),$(EXTLIB) $(call abslib, $(LIBDEP)) $(XBLASLIB)) \
		$(LDFLAGS) $(CLDFLAGS)
else
.PHONY: $(1)
endif
else
.PHONY: $(1)
endif
endef

define DEFINE_LINK_FC_RULE
ifneq (,$(strip $(FC)))
ifneq (,$(wildcard $(LIBDEP)))
ifneq (,$(wildcard $(FORTDEP)))
ifneq (,$(wildcard $(EXTDEP)))
$(1): $(2) $(call objname, $(FXXSRCS) $(F77SRCS) $(F90SRCS)) $(FORTDEP) $(LIBDEP) $(EXTDEP) $(dir $(1))/.make
	$(FC) $(SLDFLAGS) -o $(1) $(2) $(call abslib, $(FORTDEP)) $(EXTLIB) \
		$(call abslib, $(LIBDEP)) $(XBLASLIB) $(FCMTFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
.PHONY: $(1)
endif
else
.PHONY: $(1)
endif
else
.PHONY: $(1)
endif
else
.PHONY: $(1)
endif
endef

$(foreach SRC, $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX),$(eval $(call DEFINE_LINK_LD_RULE, \
  $(basename $(notdir $(SRC))), $(call objname, $(SRC)))))

$(foreach SRC, $(FXXSRCX) $(F77SRCX) $(F90SRCX),$(eval $(call DEFINE_LINK_FC_RULE, \
  $(basename $(notdir $(SRC))), $(call objname, $(SRC)))))

ifeq (0,$(MIX))
$(OUTDIR)/headeronly: $(OUTDIR)/.make $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o
	$(LD) -o $@ $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
else # compile headeronly_aux.c as C++ translation unit
$(OUTDIR)/headeronly: $(OUTDIR)/.make $(BLDDIR)/headeronly-c.o
	@ln -fs headeronly_aux.c headeronly_aux.cpp
	$(CXX) -o $@ $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) headeronly_aux.cpp \
		$(BLDDIR)/headeronly-c.o $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)
	@rm -f headeronly_aux.cpp
endif

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc #$(DEPDIR)/include/libxsmm_source.h
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

#$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

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

