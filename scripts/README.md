# Scripts

## Overview

LIBXSMM's collection of [scripts](https://github.com/libxsmm/libxsmm/tree/main/scripts) consists of Python and Bash scripts falling into two categories:

* Configuration scripts
* Development tools

Scripts related to configuring LIBXSMM are distributed with source code archives. Development tools mostly for software development purpose and are (indirectly) used by contributors, but some scripts are distributed by source code archives as well. The latter are mostly related to running tests (indirectly used by upstream maintainers, e.g., of Linux distributions).

## Configuration

Configuration scripts are usually automatically invoked by LIBXSMM's Makefile based build system (GNU Make), and there is no immediate need to run any of these scripts.

* `libxsmm_config.py`, `libxsmm_interface.py`, and `libxsmm_specialized.py`: Configures LIBXSMM and instantiates `libxsmm_version.h` (format is suitable for C/C++ and Fortran), `libxsmm_config.h`, `libxsmm.h`, and `libxsmm.f`. The [templates](https://github.com/libxsmm/libxsmm/blob/main/src/template) contain certain placeholders which are filled with actual values. Beside of the version header, the configuration considers special needs and rarely needs to deviate from the default. The default for instance allows a 3rd party build system to ease building LIBXSMM. The configuration does not consider the platform, compiler, or build system related choices but is rather about generating application specific implementations and interfaces.
* `libxsmm_dispatch.py`: Makes application specific implementations available for LIBXSMM's code registry, i.e., registers functions generated at the time of configuring and building the library.
* `libxsmm_utilities.py`: Utility functions used by other Python scripts ("library"). The script also exposes a private command-line interface to allows accessing some services, e.g., determining the name (mnemonic) of the target architecture used by LIBXSMM when JIT-generating code.
* `libxsmm_source.py`: Collects source code file names and includes these implementations when using LIBXSMM as header-only library.

Although `libxsmm_utilities.py` command-line interface is private (can change without notices), it is supposed to provide the following information:

* `libxsmm_utilities`: outputs LIBXSMM's target architecture as used by JIT code generation. For this functionality, LIBXSMM must be built since the script binds against `libxsmm_get_target_arch()`.
* `libxsmm_utilities 0`: outputs LIBXSMM's version string (preprocessor symbol `LIBXSMM_VERSION`).
* `libxsmm_utilities 1`: outputs LIBXSMM's 1st component version number (`LIBXSMM_VERSION_MAJOR`).
* `libxsmm_utilities 2`: outputs LIBXSMM's 2nd component version number (`LIBXSMM_VERSION_MINOR`).
* `libxsmm_utilities 3`: outputs LIBXSMM's 3rd component version number (`LIBXSMM_VERSION_UPDATE`).
* `libxsmm_utilities 4`: outputs LIBXSMM's 4th component version number (`LIBXSMM_VERSION_PATCH`).

The version information is based on [version.txt](https://github.com/libxsmm/libxsmm/blob/main/version.txt), which is part of LIBXSMM's source code archives (distribution).

## Development

### Overview

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
* `tool_logperf.sh`: Extracts performance information produced by certain examples (driver code), e.g., [LIBXSMM-DNN tests](https://github.com/libxsmm/libxsmm-dnn/tree/main/tests).
* `tool_logrept.sh`: Calls `tool_logperf.sh` to summarize performance, updates a database of history to generate a report (`tool_report.py`/`tool_report.sh`), and prints a base64 encoded image.
* `tool_pexec.sh`: Reads standard input and attempts to execute every line (command) on a per CPU-core basis, which can help to parallelize tests on a per-process basis.
* `tool_report.py`: Core developer team can collect a performance history of specified CI-collection (Buildkite pipeline).
* `tool_scan.sh`: Core developer team can scan the repository based on a list of keywords.
* `tool_test.sh`: 
* `tool_version.sh`: Determines LIBXSMM's version from the history of the checked-out repository (Git). With respect to LIBXSMM's patch version, the information is not fully accurate given a non-linear history.

### Parallel Execution

The script `tool_pexec.sh` can execute commands read from standard input (see `-h` or `--help`). The execution may be concurrent on a per-command basis. The level of parallelism is determined automatically but can be adjusted (oversubscription, nested parallelism). Separate logfiles can be written for every executed command (`-o /path/to/basename.ext` used as template for individual logfiles like /path/to/*basename*-case_xyz.*ext*). File I/O can become a bottleneck on distributed filesystems (e.g., NFS), or generally hinders nested parallelism (`-o /dev/null -k`).

Every line of standard input denotes a separate command:

```bash
seq 100 | xargs -I{} echo "echo \"{}\"" \
        | tool_pexec.sh
```

The script considers an allow-list which permits certain error codes. Allow-lists can be automatically generated (`-u`). Most if not all settings can be determined by environment variables as well (prefix `PEXEC_`), e.g., `export PEXEC_LG=/path/to/basename.ext` allows to omit `-o` on the command line and to always generate logfiles (see earlier explanation).

### Performance Report

The script `tool_report.py` collects performance results given in two possible formats: <span>(1)&#160;native</span> "telegram" format, and <span>(2)&#160;JSON</span> format. The script aims to avoid encoding domain knowledge. In fact, the collected information is not necessarily performance data but a time series in general. Usually, the script is not executed directly but launched using a wrapper supplying the authorization token and further adapting to the execution environment (setup):

```bash
#!/usr/bin/env bash

# authorization token
TOKEN=0123456789abcdef0123456789abcdef01234567

PYTHON=$(command -v python3)
if [ ! "${PYTHON}" ]; then
  PYTHON=$(command -v python)
fi

if [ "${PYTHON}" ]; then
  HERE=$(cd "$(dirname "$0")" && pwd -P)
  NAME=$(basename "$0" .sh)
  SCRT=${NAME}.py

  if [ "${HERE}" ]; then SCRT_A=${HERE}/${SCRT}; fi
  if [ "${LIBXSMMROOT}" ]; then SCRT_B=${LIBXSMMROOT}/scripts/${SCRT}; fi
  if [ "${REPOREMOTE}" ]; then SCRT_C=${REPOREMOTE}/libxsmm/scripts/${SCRT}; fi

  if [ "${SCRT_A}" ] && [ -e "${SCRT_A}" ]; then
    ${PYTHON} "${SCRT_A}" --token "${TOKEN}" "$@"
  elif [ "${SCRT_B}" ] && [ -e "${SCRT_B}" ]; then
    ${PYTHON} "${SCRT_B}" --token "${TOKEN}" "$@"
  elif [ "${SCRT_C}" ] && [ -e "${SCRT_C}" ]; then
    ${PYTHON} "${SCRT_C}" --token "${TOKEN}" "$@"
  else
    >&2 echo -n "ERROR: missing ${SCRT_A}"
    if [ "${SCRT_B}" ]; then >&2 echo -n " or ${SCRT_B}"; fi
    if [ "${SCRT_C}" ]; then >&2 echo -n " or ${SCRT_C}"; fi
    >&2 echo "!"
    exit 1
  fi
else
  >&2 echo "ERROR: missing prerequisites!"
  exit 1
fi
```

<a name="performance-report-flow"></a>The following flow is established:

1. Connect to a specified pipeline (online) or load a logfile directly (offline).
2. Populate an entry (JSON-block or telegram) under a "build number", "category", and "case".
3. Plot "execution time" over the history of build numbers.

There are several command-line options to customize each of the above steps (`--help` or `-h`):

* To only plot data (already collected), use `-i ""` to omit a network connection.
* To query, e.g., ResNet-50 results, use `-y resnet-50` (case-insensitive).
* Multiple results can be combined, i.e., use `-y` (space-separated words).
* To exactly match (single results) use `-x` in addition to `-y`.
* To limit and select a specific "category" (instead of all), use `-s`.
* Select exactly using `-z`, e.g., `-z -s "clx"` (omits, e.g., "clx").
* Create a PDF (vector graphics have infinite resolution), use `-g myreport.pdf`.
* Adjust pixel resolution, aspect ratio, or density, use `-d 1200x800`.

The level of verbosity (`-v`) can be adjusted (0: quiet, 1: automation, 2: progress). Default verbosity shows progress (downloading results) whereas "automation" allows to further automate reports, e.g., get the filename of the generated plot (errors are generally printed to `stderr`). Loading a logfile into the database directly can serve two purposes: <span>(1)&#160;debugging</span> the supported format like "telegram" or JSON, and <span>(2)&#160;offline</span> operation. The latter can be also useful if for instance a CI-agents produces a log, i.e., it can load into the database right away. Command-line options also allow for "exact placement" (`-j`) by specifying the build number supposed to take the loaded data (data is appended by default, i.e., it is assumed to be a new build, or the build number is incremented).

The tool automatically performs database backups of the historic values (`-n`) according to the retention (`-k`). The retention can also be used to prune the history (by forcing an earlier backup). Backups carry a timestamp as part of the filename (database), and contain the full history of value at the time of the backup.

<a name="performance-report-examples"></a>Examples (omit `-i ""` if downloading results is desired):

* Plot ResNet-50 results from CI-pipeline "tpp-libxsmm" for "clx" systems:  
  `scripts/tool_report.sh -p tpp-libxsmm -i "" -y "resnet-50" -z -s "clx"`.
* Like above request, but only FP32 results:  
  `scripts/tool_report.sh -p tpp-libxsmm -i "" -x -y "ResNet-50 (fwd, mb=1, f32)" -z -s "clx"`.
* Like above request, but alternatively ("all" operator is also default):  
  `scripts/tool_report.sh -p tpp-libxsmm -i "" -u "all" -y "resnet f32" -z -s "clx"`.
* Plot ResNet-50 results from CI-pipeline "tpp-plaidml":  
  `scripts/tool_report.sh -p tpp-plaidml -i "" -r "duration_per_example,1000,ms"`
* Plot "GFLOP/s" for "conv2d_odd_med" from CI-pipeline "tpp-plaidml":  
  `scripts/tool_report.sh -p tpp-plaidml -i "" -y "conv2d_odd_med" -r "gflop"`
* Plot "tpp-benchmark" pipeline (MLIR benchmarks, main-branch):  
  `scripts/tool_report.sh -p tpp-benchmark -i "" -y "" -r "mlir" -b "main"`
* Plot "tpp-benchmark" pipeline (MLIR benchmarks, main-branch, untied plots):  
  `scripts/tool_report.sh -p tpp-benchmark -i "" -y "" -r "mlir" -b "main" -u`
* Plot "tpp-benchmark" pipeline (reference benchmarks; selected entries `-y`):  
  `scripts/tool_report.sh -p tpp-benchmark -i "" -q "any" -y "gemm matmul" -r "dnn"`
* Plot "tpp-benchmark" pipeline (MLIR benchmarks without "single", untied plots):  
  `scripts/tool_report.sh -p tpp-benchmark -i "" -q "not" -y "single" -r "mlir" -u`
* Plot "tpp-performance" pipeline (MLIR benchmarks only "mlp", untied plots):  
  `scripts/tool_report.sh -p tpp-performance -i "" -y "mlp" -r "mlir" -u`

The exit code of the script is non-zero in case of an error, or if the latest value deviates and exceeds the margin (`--bounds`). For the latter, the meaning of the values must be given (like "higher is better"). The first argument of the bounds is a factor such that the standard deviation of historic values is amplified to act as margin of the relative deviation (latest versus previous value). The second argument of the bounds determines the accepted percentage of deviation (latest versus previous value).

The exit code is only impacted if an explicit sign is given to determine "bad" values (`+` or `-`). For example, `2.0` gives a factor of two over standard deviation (no impact for the exit code), `2.0 10` is likewise but also limits the deviation to 10% at most, `+3.0` gives a factor of three over standard deviation and considers a positive deviation as regression (like for timing values) and thereby impacts the exit code. Also, it is possible to just determine the meaning and keep default bounds, .i.e., only `-` determines negative deviation as a regression (like for higher-is-better values).

By default, only the last/current vs the previous value (history) determines the deviation. This can miss a historic slow/steady regession if the last value is always below threashold. To consider the full history value (median) as part of the deviation, a leading `0` can be given in front of the afore mentioned factor (after the sign). For example, `-02.1` specifies an aplification factor of 2.1 over standard deviation with deviation considering the median of all historic values. Consequently, a fractional factor requires two leading zeros, e.g., `-00.4`.
