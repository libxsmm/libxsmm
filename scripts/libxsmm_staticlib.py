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

    def getGenerator(self):
        return self.gen_exec


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

    def generateCode(self, outdir):
        self.logger.info(f"Starting codegen for library/export {self.name} of type {self.codetype}")
        odir = os.path.join(outdir, self.name)
        self.logger.info(f"outdir is {odir}")
        os.makedirs(odir, exist_ok=True)

        for kernel in self.kernels:
            cmd = list()
            cmd.append(self.env.getGenerator())
            cmd.append(kernel.getArch())
            cmd.append(os.path.join(odir, kernel.getName()))
            cmd.append(kernel.getType())
            cmd.extend(kernel.getConfig())
            self.logger.info(F"generator arg list: {cmd}")
            res = self.runner.run(cmd)
            self.logger.info(res.stdout)
            self.logger.error(res.stderr)

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
