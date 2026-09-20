# Copyright (c) 2015, 2016  Dave Love, University of Liverpool
# Copyright (c) 2018  Dave Love, University of Manchester
# MIT licence, per Fedora policy

# Notes:
# The specific compiler flags used are presumably chosen sensibly for the
# code, and there's no likely security implication for this.

# LIBXSMM 2.0 removes parts of the former public API and starts a new ABI
# generation.  Keep the package file list aligned with the upstream SONAME.
%global somajor 2

Name:		libxsmm
Version:	0.0.0
Release:	%autorelease
Summary:	Small dense or sparse matrix multiplications and convolutions for x86_64
License:	BSD-3-Clause
URL:		https://github.com/libxsmm/libxsmm
Source0:	https://github.com/libxsmm/libxsmm/archive/%{version}/%{name}-%{version}.tar.gz
BuildRequires:	cmake
BuildRequires:	gcc
BuildRequires:	gcc-c++
BuildRequires:	gcc-gfortran
BuildRequires:	python3-devel
ExclusiveArch:	x86_64 aarch64 riscv64 ppc64le


%description
LIBXSMM is high performance library for small dense and sparse linear
algebra opertions incl. GEMM and elementwise primities often seen in
deep learning applications. It also serves as reference implementation
of Tensor Processing Primitives (TPP), a programming abstraction for
efficient and portable deep learning and HPC workloads. With version
2.0, LIBXSMM focuses on providing a complete and architecture-portable
set of TPPs (small dense and sparse matrix operations as well as
element-wise, GEMM, and BRGEMM primitives) from which higher-level
operators such as convolutions, fully-connected layers, normalization,
and pooling are composed. LIBXSMM targets Intel Architecture with
Intel SSE, Intel AVX, Intel AVX2, Intel AVX-512 (with VNNI and
Bfloat16), and Intel AMX (Advanced Matrix Extensions), AArch64 (NEON,
SVE, and SME), RISC-V (RVV), and PowerPC 64-bit little-endian (POWER10
with VSX and MMA). Code generation is mainly based on Just-In-Time
(JIT) code specialization for compiler-independent performance (matrix
multiplications, matrix transpose/copy, sparse functionality, and
tensor primitives). LIBXSMM is suitable for "build once and deploy
everywhere", i.e., no special target flags are needed to exploit the
available performance. Supported GEMM datatypes are: FP64, FP32, FP16,
bfloat16, BF8, HF8, MXBF8, MXHF8, int16, int8, MXBF6, MXHF6, MXFP4,
int4, int2 and int1. Additionally, various non-standard low precision
combinations are supported.


%package	devel
Summary:	Development files for %name
Requires:	%name%{?_isa} = %version-%release
Requires:	pkgconfig

%description	devel
The %name-devel package contains libraries and header files for
developing applications that use %name.

%package	doc
Summary:	Documentation for %name
Requires:	%name = %version-%release
BuildArch:	noarch

%description	doc
Documentation for %name.


%prep
%autosetup -p1
# MS-Windows project files are neither useful documentation on Fedora nor
# suitable for the documentation package.
find samples -name '*.vcxproj' -delete

%conf
%cmake \
  -DCMAKE_INSTALL_Fortran_MODULES:PATH=%{_fmoddir}/%{name} \
  -DBUILD_TESTING:BOOL=ON \
  -DLIBXSMM_FORTRAN:BOOL=ON \
  -DLIBXSMM_CTEST_WITH_BLAS_REFERENCE:BOOL=OFF

%build
%cmake_build

%install
%cmake_install

%check
%ctest


%files
%license LICENSE.md
%{_libdir}/libxsmm.so.%{somajor}{,.*}
%{_libdir}/libxsmmf.so.%{somajor}{,.*}

%files devel
%{_libdir}/libxsmm.so
%{_libdir}/libxsmmf.so
%{_includedir}/%{name}/
%{_fmoddir}/%{name}/
%{_bindir}/libxsmm_gemm_generator
%{_bindir}/libxsmm_binaryexport_generator
%{_libdir}/cmake/%{name}/
%{_libdir}/pkgconfig/

%files doc
%dir %{_docdir}/%{name}
%doc %{_docdir}/%{name}/


%changelog
%autochangelog
