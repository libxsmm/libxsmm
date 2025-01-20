#!/usr/bin/env python3
# -*- coding: utf-8 -*-
###############################################################################
# Copyright (c) Intel Corporation - All rights reserved.                      #
# This file is part of the LIBXSMM library.                                   #
#                                                                             #
# For information on the license, see the LICENSE file.                       #
# Further information: https://github.com/libxsmm/libxsmm/                    #
# SPDX-License-Identifier: BSD-3-Clause                                       #
###############################################################################
# Alexander Heinecke (Intel Corp.), Renato Golin (Intel Corp.)
###############################################################################

"""
    Script to use libxsmm's binaryexport driver to create static elf objects and/or
    a staic library and/or C files with inline assembly.
    The statically generated code snippets are specified via JSON files

    The JSON format is:
[
  {
    "libmykernels": {
      "type" : "library",
      "kernels": {
        "one": {
          "type" : "gemm",
          "arch" : "hsw",
          "config" : [ "F32", "F32", "F32", "F32", "32", "32", "32", "32", "32", "32", "1", "1", "0", "0", "0", "0", "0", "0", "0", "nopf", "nobr", "1", "0", "0", "0", "0" ]
        },
        "two": {
          "type" : "gemm",
          "arch" : "hsw",
          "config" : [ "F32", "F32", "F32", "F32", "32", "32", "32", "32", "32", "32", "1", "1", "0", "0", "0", "0", "0", "0", "0", "nopf", "strdbr", "1", "0", "4096", "4096", "0" ]
        }
      }
    }
  }
]
"""

import os
import sys
import re
import argparse
import json
import shlex
import shutil
import logging
import coloredlogs
import subprocess

class Logger(object):
    def __init__(self, name, verbosity):
        self.logger = logging.getLogger(name)

        # Default level is WARNING (no output other than warnings and errors)
        start = logging.WARNING
        silent = min(verbosity * 10, logging.INFO)
        coloredlogs.install(level=start - silent, logger=self.logger)

    def error(self, err):
        self.logger.error(err)

    def warning(self, warning):
        self.logger.warning(warning)

    def info(self, info):
        self.logger.info(info)

    def debug(self, trace):
        self.logger.debug(trace)

    def silent(self):
        return self.logger.getEffectiveLevel() > logging.INFO


class Execute(object):
    """Executes commands, returns out/err"""

    def __init__(self, loglevel):
        self.logger = Logger("execute", loglevel)

    def run(self, program, input=""):
        """Execute Commands, return out/err"""

        if program and not isinstance(program, list):
            raise TypeError("Program needs to be a list of arguments")
        if not program:
            raise ValueError("Need program arguments to execute")

        if self.logger:
            self.logger.debug(f"Executing: {' '.join(program)}")

        # Call the program, capturing stdout/stderr
        result = subprocess.run(
            program,
            input=input if input else None,
            capture_output=True,
            encoding="utf-8",
        )

        # Collect stdout, stderr as UTF-8 strings
        result.stdout = str(result.stdout)
        result.stderr = str(result.stderr)

        # Return
        return result


class Environment(object):
    def __init__(self, args, loglevel):
        self.logger = Logger("EnvHelper", loglevel)
        #helper = TPPHelper(loglevel)
        self.exec_dir = os.path.realpath(os.getcwd())
        self.script_dir = os.path.realpath(os.path.dirname(__file__))
        self.root_dir = os.path.dirname(self.script_dir)
        self.logger.info(f"running out of directory: {self.exec_dir}")
        self.logger.info(f"script directory: {self.script_dir}")
        self.logger.info(f"LIBXSMM root directory: {self.root_dir}")
        self.gen_exec = os.path.join(self.root_dir, "bin", "libxsmm_binaryexport_generator")
        # check if we have llvm-objcopy avaialble
        self.llvm_objcopy = shutil.which('llvm-objcopy')
        if ( self.llvm_objcopy is None ):
            self.logger.error(f"didn't find llvm-objcopy, we cannot proceed :-(")
            sys.exit(1)
        else:
            self.logger.info(f"found llvm-objcopy {self.llvm_objcopy}")
        # check if we have ar available
        self.ar = shutil.which('ar')
        if ( self.ar is None ):
            self.logger.error(f"didn't find ar, we cannot proceed :-(")
            sys.exit(1)
        else:
            self.logger.info(f"found ar {self.ar}")
        # check if we have rm available
        self.rm = shutil.which('rm')
        if ( self.rm is None ):
            self.logger.error(f"didn't find rm, we cannot proceed :-(")
            sys.exit(1)
        else:
            self.logger.info(f"found ar {self.rm}")

    def getGenerator(self):
        return self.gen_exec

    def getLlvmObjcopy(self):
        return self.llvm_objcopy

    def getAr(self):
        return self.ar

    def getRm(self):
        return self.rm


class KernelDesc(object):
    # Class to hold GEMM kernel configuration
    def __init__(self, loglevel, name, type, arch, config):
        self.logger = Logger("GemmKernel", loglevel)
        self.name = name
        self.arch = arch
        self.type = type
        self.config = config
        self.logger.info(f"created a new KernelDesc Object with type {self.type} and name {self.name} for arch {self.arch} with config {self.config}")

    def getArch(self):
        return self.arch

    def getName(self):
        return self.name

    def getType(self):
        return self.type

    def getConfig(self):
        return self.config


class CodeGenKernels(object):
    # Hold the code gen properties
    def __init__(self, name, loglevel):
        self.name = name
        self.codetype = ""
        self.logger = Logger("CodeGenKernels", loglevel)
        self.runner = Execute(loglevel)
        self.env = Environment(args, loglevel)
        self.kernels = list()

    def addCodeGenKernel(self, name, json):
        kernelType = json["type"]
        self.logger.info(f"Adding {kernelType} to library {self.name}")
        if kernelType == "gemm" or kernelType == "gemmext":
            self.kernels.append(KernelDesc(loglevel, name, kernelType, json["arch"], json["config"]))
        else:
            self.logger.error(f"Unknown kerneltype '{kernelType}'")
            return False
        return True

    def setCodeType(self, codetype):
        self.codetype = codetype

    def getGeneratorConfigString(self, kernel):
        ret = os.path.basename(self.env.getGenerator()) + " "
        ret = ret + kernel.getArch() + " "
        ret = ret + kernel.getName() + " "
        ret = ret + kernel.getType() + " "
        ret = ret + json.dumps(kernel.getConfig()).strip('"')
        ret = ret.replace("[", "")
        ret = ret.replace("]", "")
        ret = ret.replace(",", "")
        ret = ret.replace("\"", "")
        return ret

    def getGeneratorCmd(self, kernel, kernelfile):
        cmd = list()
        cmd.append(self.env.getGenerator())
        cmd.append(kernel.getArch())
        cmd.append(kernelfile)
        cmd.append(kernel.getType())
        cmd.extend(kernel.getConfig())
        return cmd

    def getObjcopyBintoElfCmd(self, kernel, kernelfile):
        cmd = list()
        cmd.append(self.env.getLlvmObjcopy())
        cmd.append("-I")
        cmd.append("binary")
        cmd.append("-O")
        cmd.append("elf64-x86-64")
        cmd.append(f"--rename-section=.data=.text.{kernel.getName()},code")
        cmd.append(kernelfile)
        cmd.append(kernelfile + ".tmp.elf")
        return cmd

    def getObjcopyElftoFinalCmd(self, kernel, kernelfile):
        file = kernelfile.replace("/", "_")
        file = file.replace(".", "_")
        cmd = list()
        cmd.append(self.env.getLlvmObjcopy())
        cmd.append("-I")
        cmd.append("elf64-x86-64")
        cmd.append("-O")
        cmd.append("elf64-x86-64")
        cmd.append(f"--redefine-sym=_binary_{file}_start={kernel.getName()}")
        cmd.append(f"--redefine-sym=_binary_{file}_end={kernel.getName()}_end")
        cmd.append(f"--redefine-sym=_binary_{file}_size={kernel.getName()}_size")
        cmd.append(kernelfile + ".tmp.elf")
        cmd.append(kernelfile + ".o")
        return cmd

    def getRmCmd(self, kernelfile):
        cmd = list()
        cmd.append(self.env.getRm())
        cmd.append("-f")
        cmd.append(kernelfile + ".tmp.elf")
        return cmd

    def getArCmd(self, archivename, objfiles):
        cmd = list()
        cmd.append(self.env.getAr())
        cmd.append("rcs")
        cmd.append(archivename)
        cmd.extend(objfiles)
        return cmd

    def generateCode(self, outdir):
        objfiles = list()
        self.logger.info(f"Starting codegen for library/export {self.name} of type {self.codetype}")
        odir = os.path.join(outdir, self.name)
        self.logger.info(f"outdir is {odir}")
        os.makedirs(odir, exist_ok=True)
        headerfile = open(os.path.join(odir, self.name + ".h"), "w")
        self.logger.info(f"opened headerfile for writing: {headerfile.name}")
        headerfile.write("#ifndef " + self.name.upper() + "_H\n")
        headerfile.write("#define " + self.name.upper() + "_H\n\n")
        headerfile.write("#ifdef __cplusplus\n")
        headerfile.write("extern \"C\" {\n")
        headerfile.write("#endif\n\n")
        headerfile.write("typedef struct libxsmm_matrix_op_arg {\n")
        headerfile.write("  void* primary;\n")
        headerfile.write("  void* secondary;\n")
        headerfile.write("  void* tertiary;\n")
        headerfile.write("  void* quaternary;\n")
        headerfile.write("} libxsmm_matrix_op_arg;\n\n")
        headerfile.write("typedef struct libxsmm_matrix_arg {\n")
        headerfile.write("  void* primary;\n")
        headerfile.write("  void* secondary;\n")
        headerfile.write("  void* tertiary;\n")
        headerfile.write("  void* quaternary;\n")
        headerfile.write("} libxsmm_matrix_arg;\n\n")
        headerfile.write("typedef struct libxsmm_gemm_param {\n")
        headerfile.write("  libxsmm_matrix_op_arg op;\n")
        headerfile.write("  libxsmm_matrix_arg a;\n")
        headerfile.write("  libxsmm_matrix_arg b;\n")
        headerfile.write("  libxsmm_matrix_arg c;\n")
        headerfile.write("} libxsmm_gemm_param;\n\n")

        for kernel in self.kernels:
            # generate binary code
            kernelfile = os.path.join(odir, kernel.getName())
            gencmd = self.getGeneratorCmd(kernel, kernelfile)
            self.logger.info(F"generator arg list: {gencmd}")
            res = self.runner.run(gencmd)
            self.logger.info(res.stdout)
            if (res.stderr != ""):
                 self.logger.error(res.stderr)
            # create elf object from binary code
            llvmobjcopy_one_cmd = self.getObjcopyBintoElfCmd(kernel, kernelfile)
            self.logger.info(F"llvm-objcopy tmp cmd: {llvmobjcopy_one_cmd}")
            res = self.runner.run(llvmobjcopy_one_cmd)
            self.logger.info(res.stdout)
            if (res.stderr != ""):
                 self.logger.error(res.stderr)
                 sys.exit(1)

            llvmobjcopy_two_cmd = self.getObjcopyElftoFinalCmd(kernel, kernelfile)
            self.logger.info(F"llvm-objcopy final cmd: {llvmobjcopy_two_cmd}")
            res = self.runner.run(llvmobjcopy_two_cmd)
            self.logger.info(res.stdout)
            if (res.stderr != ""):
                 self.logger.error(res.stderr)
                 sys.exit(1)

            rm_cmd = self.getRmCmd(kernelfile)
            self.logger.info(F"rm cmd: {rm_cmd}")
            res = self.runner.run(rm_cmd)
            self.logger.info(res.stdout)
            if (res.stderr != ""):
                 self.logger.error(res.stderr)
                 sys.exit(1)

            objfiles.append(kernelfile + ".o")
            # add generated kernel to header
            headerfile.write("/* " + self.getGeneratorConfigString(kernel) + " */\n")
            headerfile.write("void " + kernel.getName() + "( libxsmm_gemm_param* param );\n")

        archivefile = os.path.join(odir, self.name + ".a")
        ar_cmd = self.getArCmd(archivefile, objfiles)
        self.logger.info(F"ar cmd: {ar_cmd}")
        res = self.runner.run(ar_cmd)
        self.logger.info(res.stdout)
        if (res.stderr != ""):
             self.logger.error(res.stderr)
             sys.exit(1)

        headerfile.write("\n#ifdef __cplusplus\n")
        headerfile.write("extern }\n")
        headerfile.write("#endif\n\n")
        headerfile.write("#endif /* " + self.name.upper() + "_H */\n")
        headerfile.close()

        return True


class CodeGenDriver(object):
    #Run CodeGen based on JSON configurations
    def __init__(self, args, loglevel):
        self.logger = Logger("CodeGenDriver", loglevel)
        self.generator = list()
        self.loglevel = loglevel
        self.args = args

    def createCodeGenInput(self):
        #Reads each JSON file and create a list with all runs
        # Find and read the JSON file
        configs = self.args.config.split(",")
        for config in configs:
            # Error on specific files when invalid
            if not os.path.exists(config):
                self.logger.error(
                    f"JSON config '{self.args.config}' does not exist"
                )
                raise SyntaxError("Cannot find JSON config")

            self.logger.info(f"Reading up '{config}'")
            with open(config) as jsonFile:
                jsonCfg = json.load(jsonFile)

            # Parse and add all runs
            for cfg in jsonCfg:
                if len(cfg.keys()) > 1:
                    self.logger.error(
                        "List of dict with a single element expected"
                    )
                    return False

                name = list(cfg.keys())[0]
                libconfig = cfg[name]
                codegen = CodeGenKernels(name, loglevel)
                #benchs = Benchmark(name, self.args, self.env, self.loglevel)
                for key, value in libconfig.items():
                    if key == "type":
                        codegen.setCodeType(value)
                    elif key == "kernels":
                        kernels = cfg[name]["kernels"]
                        for name, kernel in kernels.items():
                            codegen.addCodeGenKernel(name, kernel)
                    else:
                        self.logger.error(f"unkone ket type {key}")
                        return False
                self.generator.append(codegen)
        return True

    def generate(self):
        # Actually run the file in benchmark mode, no output
        self.logger.info("Starting CodeGeneration Phase")
        outdir = os.path.realpath(args.outdir)

        # loop over input files
        for gentask in self.generator:
            if not gentask.generateCode(outdir):
                return False
        return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LIBXSMM Static Code Generator Driver Utility")

    # Required argument: code gen config
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="codegen.json",
        help="JSON file containing codegen configuration",
    )

    # Required argument: code gen config
    parser.add_argument(
        "-o",
        "--outdir",
        type=str,
        default=".",
        help="output directory of the generated code",
    )

    # Optional
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="The verbosity of logging output",
    )
    args = parser.parse_args()

    # Creates the logger object
    loglevel = args.verbose
    logger = Logger("driver", loglevel)

    # Creates a controller from command line arguments
    driver = CodeGenDriver(args, loglevel)

    # Detects all benchmarks to run, validates files / args
    if not driver.createCodeGenInput():
        logger.error("Error finding Config")
        parser.print_help()
        sys.exit(1)

    # Runs all benchmarks
    if not driver.generate():
        logger.error("Error generating Code")
        sys.exit(1)
