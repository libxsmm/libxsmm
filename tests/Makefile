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
DFLAGS = $(NULL)
IFLAGS = -I$(INCDIR) -I$(DEPDIR)/include -I$(DEPDIR)/src

BLAS ?= 1
OMP ?= 1
SYM ?= 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

OUTNAME := $(shell basename $(ROOTDIR))
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh)
CPPSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CPPSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cpp 2> /dev/null | tr "\n" " ")
CXXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CXXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cxx 2> /dev/null | tr "\n" " ")
CCXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CCXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cc  2> /dev/null | tr "\n" " ")
CSOURCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
CSOURCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.c   2> /dev/null | tr "\n" " ")
FXXSRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
FXXSRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f   2> /dev/null | tr "\n" " ")
F77SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F77SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.F   2> /dev/null | tr "\n" " ")
F90SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ") \
           $(shell grep -L $(FMAIN) $(SRCDIR)/*.F90 2> /dev/null | tr "\n" " ")
F90SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f90 2> /dev/null | tr "\n" " ") \
           $(shell grep -l $(FMAIN) $(SRCDIR)/*.F90 2> /dev/null | tr "\n" " ")
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

TEST_NOBLASLIB = $(call abslib, $(LIBNAME)noblas.$(LIBEXT))
#ifneq (Darwin,$(UNAME))
  TEST_EXTLIB = $(EXTLIB)
#endif

define DEFINE_LINK_LD_RULE
ifneq (,$(wildcard $(LIBNAME).$(LIBEXT)))
ifneq (,$(wildcard $(LIBNAME)ext.$(LIBEXT)))
$(1): $(2) $(call objname, $(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS)) $(LIBNAME).$(LIBEXT) $(LIBNAME)ext.$(LIBEXT) $(dir $(1))/.make
ifneq (0,$(BLAS))
	$(LD) $(SLDFLAGS) -o $(1) $(2) $(TEST_EXTLIB) $(call abslib, $(LIBNAME).$(LIBEXT)) $(LDFLAGS) $(CLDFLAGS)
else
	$(LD) $(SLDFLAGS) -o $(1) $(2) $(call abslib, $(LIBNAME).$(LIBEXT)) $(TEST_EXTLIB) $(TEST_NOBLASLIB) $(LDFLAGS) $(CLDFLAGS)
endif
else
.PHONY: $(1)
endif
else
.PHONY: $(1)
endif
endef

define DEFINE_LINK_FC_RULE
ifneq (,$(strip $(FC)))
ifneq (,$(wildcard $(LIBNAME).$(LIBEXT)))
ifneq (,$(wildcard $(LIBNAME)f.$(LIBEXT)))
ifneq (,$(wildcard $(LIBNAME)ext.$(LIBEXT)))
$(1): $(2) $(call objname, $(FXXSRCS) $(F77SRCS) $(F90SRCS)) $(LIBNAME)f.$(LIBEXT) $(LIBNAME).$(LIBEXT) $(LIBNAME)ext.$(LIBEXT) $(dir $(1))/.make
ifneq (0,$(BLAS))
	$(FC) $(SLDFLAGS) -o $(1) $(2) $(call abslib, $(LIBNAME)f.$(LIBEXT)) $(TEST_EXTLIB) \
		$(call abslib, $(LIBNAME).$(LIBEXT)) $(FCMTFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
else
	$(FC) $(SLDFLAGS) -o $(1) $(2) $(call abslib, $(LIBNAME)f.$(LIBEXT)) $(call abslib, $(LIBNAME).$(LIBEXT)) $(TEST_EXTLIB) \
		$(TEST_NOBLASLIB) $(FCMTFLAGS) $(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS)
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
else
.PHONY: $(1)
endif
endef

$(foreach SRC, $(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX),$(eval $(call DEFINE_LINK_LD_RULE, \
  $(basename $(notdir $(SRC))), $(call objname, $(SRC)))))

$(foreach SRC, $(FXXSRCX) $(F77SRCX) $(F90SRCX),$(eval $(call DEFINE_LINK_FC_RULE, \
  $(basename $(notdir $(SRC))), $(call objname, $(SRC)))))

$(OUTDIR)/headeronly: $(OUTDIR)/.make $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o
	$(LD) -o $@ $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o $(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS)

$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(CC) $(CFLAGS) $(DFLAGS) $(IFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make $(ROOTDIR)/Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(FC) $(FCMTFLAGS) $(FCFLAGS) $(DFLAGS) $(IFLAGS) $(FTARGET) -c $< -o $@

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

