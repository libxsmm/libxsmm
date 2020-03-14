ROOTDIR = $(abspath $(dir $(firstword $(MAKEFILE_LIST))))
DEPDIR = ..
SRCDIR = .
INCDIR = $(ROOTDIR)
BLDDIR = build
OUTDIR = .

CXXFLAGS = $(NULL)
CFLAGS = $(NULL)
DFLAGS = -DLIBXSMM_BLAS_CONST

# PEDANTIC=2: OpenBLAS headers can cause warnings
override PEDANTIC = 1
BLAS ?= 1
OMP ?= 1
SYM ?= 1

# include common Makefile artifacts
include $(DEPDIR)/Makefile.inc

# necessary include directories
IFLAGS += -I$(call quote,$(INCDIR))
IFLAGS += -I$(call quote,$(DEPDIR)/include)
#IFLAGS += -I$(call quote,$(DEPDIR)/src)

ifneq (0,$(shell echo "$$((100000>$(GCC_VERSION_NUM)))"))
  MIX ?= 1
else
  MIX ?= 0
endif

OUTNAME := $(shell basename "$(ROOTDIR)")
HEADERS := $(wildcard $(INCDIR)/*.h) $(wildcard $(INCDIR)/*.hpp) $(wildcard $(INCDIR)/*.hxx) $(wildcard $(INCDIR)/*.hh) \
           $(wildcard $(SRCDIR)/*.h) $(wildcard $(SRCDIR)/*.hpp) $(wildcard $(SRCDIR)/*.hxx) $(wildcard $(SRCDIR)/*.hh) \
           $(DEPDIR)/include/libxsmm_source.h
CPPSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cpp 2>/dev/null | tr -s "\n" " ")
CPPSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cpp 2>/dev/null | tr -s "\n" " ")
CXXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cxx 2>/dev/null | tr -s "\n" " ")
CXXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cxx 2>/dev/null | tr -s "\n" " ")
CCXSRCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.cc  2>/dev/null | tr -s "\n" " ")
CCXSRCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.cc  2>/dev/null | tr -s "\n" " ")
CSOURCS := $(shell grep -L $(CMAIN) $(SRCDIR)/*.c   2>/dev/null | tr -s "\n" " ")
CSOURCX := $(shell grep -l $(CMAIN) $(SRCDIR)/*.c   2>/dev/null | tr -s "\n" " ")
FXXSRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f   2>/dev/null | tr -s "\n" " ")
FXXSRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f   2>/dev/null | tr -s "\n" " ")
F77SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.F   2>/dev/null | tr -s "\n" " ")
F77SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.F   2>/dev/null | tr -s "\n" " ")
F90SRCS := $(shell grep -L $(FMAIN) $(SRCDIR)/*.f90 2>/dev/null | tr -s "\n" " ") \
           $(shell grep -L $(FMAIN) $(SRCDIR)/*.F90 2>/dev/null | tr -s "\n" " ")
F90SRCX := $(shell grep -l $(FMAIN) $(SRCDIR)/*.f90 2>/dev/null | tr -s "\n" " ") \
           $(shell grep -l $(FMAIN) $(SRCDIR)/*.F90 2>/dev/null | tr -s "\n" " ")
MODULES := $(addsuffix    .mod,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS))) \
           $(addsuffix .modmic,$(basename $(FXXSRCS) $(F77SRCS) $(F90SRCS)))
OBJECTS := $(call objname,$(CPPSRCS) $(CXXSRCS) $(CCXSRCS) $(CSOURCS))
OBJECTX := $(call objname,$(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX))
FTNOBJS := $(call objname,$(FXXSRCS) $(F77SRCS) $(F90SRCS))
FTNOBJX := $(call objname,$(FXXSRCX) $(F77SRCX) $(F90SRCX))
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
$(1): $(2) $(OBJECTX) $(dir $(1))/.make $(if $(filter $(1),$(HEADER_ONLY)),$(NOBLASDEP),$(LIBDEP) $(EXTDEP))
	$(LD) $(SLDFLAGS) -o $(1) $(2) \
		$(if $(filter $(1),$(HEADER_ONLY)),$(NOBLASLIB),$(MAINLIB) $(EXTLIB)) \
		$(call cleanld,$(LDFLAGS) $(CLDFLAGS))
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
$(1): $(2) $(FTNOBJS) $(FORTDEP) $(LIBDEP) $(EXTDEP) $(dir $(1))/.make
	$(FC) $(SLDFLAGS) -o $(1) $(2) $(MAINLIB) $(FORTLIB) $(EXTLIB) \
		$(FCMTFLAGS) $(call cleanld,$(LDFLAGS) $(FLDFLAGS) $(ELDFLAGS))
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

$(foreach SRC, $(filter-out $(SRCDIR)/headeronly.c,$(CPPSRCX) $(CXXSRCX) $(CCXSRCX) $(CSOURCX)), \
  $(eval $(call DEFINE_LINK_LD_RULE, $(basename $(notdir $(SRC))), $(call objname,$(SRC)))))

$(foreach SRC, $(FXXSRCX) $(F77SRCX) $(F90SRCX), \
  $(eval $(call DEFINE_LINK_FC_RULE, $(basename $(notdir $(SRC))), $(call objname,$(SRC)))))

ifeq (0,$(MIX))
$(OUTDIR)/headeronly: $(OUTDIR)/.make $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o $(NOBLASDEP)
	$(LD) -o $@ $(BLDDIR)/headeronly-c.o $(BLDDIR)/headeronly_aux-c.o \
		$(call cleanld,$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(NOBLASLIB))
else # compile headeronly_aux.c as C++ translation unit
$(OUTDIR)/headeronly: $(OUTDIR)/.make $(BLDDIR)/headeronly-c.o $(NOBLASDEP)
	@$(CP) $(SRCDIR)/headeronly_aux.c $(SRCDIR)/headeronly_aux.cpp
	$(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $(SRCDIR)/headeronly_aux.cpp -o $(BLDDIR)/headeronly_aux-cpp.o
	$(XLD) -o $@ $(BLDDIR)/headeronly_aux-cpp.o $(BLDDIR)/headeronly-c.o \
		$(call cleanld,$(SLDFLAGS) $(LDFLAGS) $(CLDFLAGS) $(NOBLASLIB))
	@rm -f headeronly_aux.cpp
endif

#$(BLDDIR)/%-cpp.o: $(SRCDIR)/%.cpp .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc #$(DEPDIR)/include/libxsmm_source.h
# $(CXX) $(DFLAGS) $(IFLAGS) $(CXXFLAGS) $(CTARGET) -c $< -o $@

$(BLDDIR)/%-c.o: $(SRCDIR)/%.c .state $(BLDDIR)/.make $(HEADERS) Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
	$(CC) $(DFLAGS) $(IFLAGS) $(CFLAGS) $(CTARGET) -c $< -o $@

#$(BLDDIR)/%-f.o: $(SRCDIR)/%.f .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f90.o: $(SRCDIR)/%.f90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f90.o: $(SRCDIR)/%.F90 .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

#$(BLDDIR)/%-f77.o: $(SRCDIR)/%.F .state $(BLDDIR)/.make Makefile $(DEPDIR)/Makefile.inc $(DEPDIR)/include/libxsmm_source.h
#$(FC) $(FCMTFLAGS) $(DFLAGS) $(IFLAGS) $(FCFLAGS) $(FTARGET) -c $< -o $@

.PHONY: clean
clean:
ifneq ($(call qapath,$(BLDDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(BLDDIR)),$(call qapath,.))
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
ifneq ($(call qapath,$(OUTDIR)),$(ROOTDIR))
ifneq ($(call qapath,$(OUTDIR)),$(call qapath,.))
	@rm -rf $(OUTDIR)
endif
endif
ifneq (,$(wildcard $(OUTDIR))) # still exists
	@rm -f $(OUTDIR)/libxsmm.$(DLIBEXT) $(OUTDIR)/*.stackdump
	@rm -f $(XFILES) $(MODULES)
endif

