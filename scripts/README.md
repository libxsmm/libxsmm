## Scripts

### Overview

LIBXSMM's collection of [scripts](https://github.com/libxsmm/libxsmm/tree/main/scripts) consists of Python and Bash scripts falling into two categories:

* Configuration scripts
* Development tools

Scripts related to configuring LIBXSMM are distributed with source code archives. Development tools mostly for software development purpose and are (indirectly) used by contributors, but some scripts are distributed by source code archives as well. The latter are mostly related to running tests (indirectly used by upstream maintainers, e.g., of Linux distributions).

### Configuration Scripts

Configuration scripts are usually automatically invoked by LIBXSMM's Makefile based build system (GNU Make), and there is no immediate need to run any of these scripts.

* `libxsmm_config.py`, `libxsmm_interface.py`, and `libxsmm_specialized.py`: Configures LIBXSMM and instantiates `libxsmm_version.h` (format is suitable for C/C++ and Fortran), `libxsmm_config.h`, `libxsmm.h`, and `libxsmm.f`. The [templates](https://github.com/libxsmm/libxsmm/blob/main/src/template) contain certain placeholders which are filled with actual values. Beside of the version header, the configuration considers special needs and rarely needs to deviate from the default. The default for instance allows a 3rd party build system to ease building LIBXSMM. The configuration does not consider the platform, compiler, or build system related choices but is rather about generating application specific implementations and interfaces.
* `libxsmm_dispatch.py`: Makes application specific implementations available for LIBXSMM's code registry, i.e., registers functions generated at the time of configuring and building the library.
* `libxsmm_utilities.py`: Utility functions used by other Python scripts ("library"). The script also exposes a private command line interface to allows accessing some services, e.g., determining the name (mnemonic) of the target architecture used by LIBXSMM when JIT-generating code.
* `libxsmm_source.py`: Collects source code file names and includes these implementations when using LIBXSMM as header-only library.

Although `libxsmm_utilities.py` command line interface is private (can change without notices), it is supposed to provide the following information:

* `libxsmm_utilities`: outputs LIBXSMM's target architecture as used by JIT code generation. For this functionality, LIBXSMM must be built since the script binds against `libxsmm_get_target_arch()`.
* `libxsmm_utilities 0`: outputs LIBXSMM's version string (preprocessor symbol `LIBXSMM_VERSION`).
* `libxsmm_utilities 1`: outputs LIBXSMM's 1st component version number (`LIBXSMM_VERSION_MAJOR`).
* `libxsmm_utilities 2`: outputs LIBXSMM's 2nd component version number (`LIBXSMM_VERSION_MINOR`).
* `libxsmm_utilities 3`: outputs LIBXSMM's 3rd component version number (`LIBXSMM_VERSION_UPDATE`).
* `libxsmm_utilities 4`: outputs LIBXSMM's 4th component version number (`LIBXSMM_VERSION_PATCH`).

The version information is based on [version.txt](https://github.com/libxsmm/libxsmm/blob/main/version.txt), which is part of LIBXSMM's source code archives (distribution).

### Development Tools

* `tool_analyze.sh`: Runs compiler based static analysis based on Clang or GCC.
* `tool_changelog.sh`: Rephrases the history of LIBXSMM's checked-out repository to consist as a changelog grouped by contributors.
* `tool_checkabi.sh`: Extracts exported/visible functions and other symbols (public interface) from built LIBXSMM and compares against a recorded state. The purpose is to acknowledge and confirm for instance removed functionality (compatibility). This includes functions only exported to allow interaction between LIBXSMM's different libraries. However, it currently falls short of recognizing changes to the signature of functions (arguments).
* `tool_cpuinfo.sh`: Informs about the system the script is running on, i.e., the number of CPU sockets (packages), the number of CPU cores, the number of CPU threads, the number of threads per CPU core (SMT), and the number of NUMA domains. The script is mainly used to parallelize tests during development. However, this script is distributed because test related scripts are not only of contributor's interest (`tool_test.sh`).
* `tool_envrestore.sh`: Restores environment variables when running tests (`tool_test.sh`).
* `tool_getenvars.sh`: Attempts to collect environment variables used in LIBXSMM's code base (`getenv`). This script is distributed.
* `tool_gitaddforks.sh`: Collects forks of LIBXSMM and adds them as Git-remotes, which can foster collaboration (development).
* `tool_gitauthors.sh`: Collects authors of LIBXSMM from history of the checked-out repository.
* `tool_gitprune.sh`: Performs garbage collection of the checked-out repository (`.git folder`). The script does not remove files, i.e., it does not run `git clean`.
* `tool_inspector.sh`: Wrapper script when running a binary to detect potential memory leaks or data races.
* `tool_normalize.sh`: Detects simple code patters banned from LIBXSMM's source code.
* `tool_perflog.sh`: Extracts performance information produced by certain examples (driver code), e.g., [LIBXSMM-DNN tests](https://github.com/libxsmm/libxsmm-dnn/tree/main/tests).
* `tool_pexec.sh`: Reads standard input and attempts to execute every line (command) on a per CPU-core basis, which can help to parallelize tests on a per-process basis.
* `tool_report.py`: Core developer team can collect a performance history of a certain CI-collection (Buildkite pipeline).
* `tool_scan.sh`: Core developer team can scan the repository based on a list of keywords.
* `tool_test.sh`: 
* `tool_version.sh`: Determines LIBXSMM's version from the history of the checked-out repository (Git). With respect to LIBXSMM's patch version, the information is not fully accurate given a non-linear history.
