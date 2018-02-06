/******************************************************************************
** Copyright (c) 2018, Intel Corporation                                     **
** All rights reserved.                                                      **
**                                                                           **
** Redistribution and use in source and binary forms, with or without        **
** modification, are permitted provided that the following conditions        **
** are met:                                                                  **
** 1. Redistributions of source code must retain the above copyright         **
**    notice, this list of conditions and the following disclaimer.          **
** 2. Redistributions in binary form must reproduce the above copyright      **
**    notice, this list of conditions and the following disclaimer in the    **
**    documentation and/or other materials provided with the distribution.   **
** 3. Neither the name of the copyright holder nor the names of its          **
**    contributors may be used to endorse or promote products derived        **
**    from this software without specific prior written permission.          **
**                                                                           **
** THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS       **
** "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT         **
** LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR     **
** A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT      **
** HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,    **
** SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED  **
** TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR    **
** PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF    **
** LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING      **
** NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS        **
** SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.              **
******************************************************************************/
/* Hans Pabst (Intel Corp.)
******************************************************************************/
#if defined(__PYTHON) && defined(LIBXSMM_BUILD) && !defined(__STATIC)
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(push,target(LIBXSMM_OFFLOAD_TARGET))
#endif
#include <Python.h> /* must be included first */
#if defined(LIBXSMM_OFFLOAD_TARGET)
# pragma offload_attribute(pop)
#endif
#endif
#include <libxsmm.h>


#if defined(__PYTHON) && defined(LIBXSMM_BUILD) && !defined(__STATIC)

LIBXSMM_API PyObject* libxsmmpy_get_target_arch(PyObject* self, PyObject* args);
LIBXSMM_API_DEFINITION PyObject* libxsmmpy_get_target_arch(PyObject* self, PyObject* args)
{
  LIBXSMM_UNUSED(self); LIBXSMM_UNUSED(args);
  return PyString_InternFromString(libxsmm_get_target_arch());
}

LIBXSMM_API PyObject* libxsmmpy_set_target_arch(PyObject* self, PyObject* args);
LIBXSMM_API_DEFINITION PyObject* libxsmmpy_set_target_arch(PyObject* self, PyObject* args)
{
  int ivalue = LIBXSMM_TARGET_ARCH_UNKNOWN;
  char* svalue = NULL;
  LIBXSMM_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "s", &svalue)) {
    libxsmm_set_target_arch(svalue);
  }
  else if (0 != PyArg_ParseTuple(args, "i", &ivalue)) {
    libxsmm_set_target_archid(ivalue);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


LIBXSMM_API PyObject* libxsmmpy_get_target_archid(PyObject* self, PyObject* args);
LIBXSMM_API_DEFINITION PyObject* libxsmmpy_get_target_archid(PyObject* self, PyObject* args)
{
  LIBXSMM_UNUSED(self); LIBXSMM_UNUSED(args);
  return Py_BuildValue("i", libxsmm_get_target_archid());
}

LIBXSMM_API PyObject* libxsmmpy_set_target_archid(PyObject* self, PyObject* args);
LIBXSMM_API_DEFINITION PyObject* libxsmmpy_set_target_archid(PyObject* self, PyObject* args)
{
  int value = LIBXSMM_TARGET_ARCH_UNKNOWN;
  LIBXSMM_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "i", &value)) {
    libxsmm_set_target_archid(value);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


LIBXSMM_API PyObject* libxsmmpy_get_verbosity(PyObject* self, PyObject* args);
LIBXSMM_API_DEFINITION PyObject* libxsmmpy_get_verbosity(PyObject* self, PyObject* args)
{
  LIBXSMM_UNUSED(self); LIBXSMM_UNUSED(args);
  return Py_BuildValue("i", libxsmm_get_verbosity());
}

LIBXSMM_API PyObject* libxsmmpy_set_verbosity(PyObject* self, PyObject* args);
LIBXSMM_API_DEFINITION PyObject* libxsmmpy_set_verbosity(PyObject* self, PyObject* args)
{
  int value = 0;
  LIBXSMM_UNUSED(self);
  if (0 != PyArg_ParseTuple(args, "i", &value)) {
    libxsmm_set_verbosity(value);
  }
  else { /* error */
    return NULL;
  }
  Py_RETURN_NONE;
}


PyMODINIT_FUNC initlibxsmm(void);
PyMODINIT_FUNC initlibxsmm(void)
{
  static PyMethodDef pymethod_def[] = {
    { "GetTargetArch", libxsmmpy_get_target_arch, METH_NOARGS,
      PyDoc_STR("Get the name of the code path.") },
    { "SetTargetArch", libxsmmpy_set_target_arch, METH_VARARGS,
      PyDoc_STR("Set the name of the code path.") },
    { "GetTargetArchId", libxsmmpy_get_target_archid, METH_NOARGS,
      PyDoc_STR("Get the id of the code path.") },
    { "SetTargetArchId", libxsmmpy_set_target_archid, METH_VARARGS,
      PyDoc_STR("Set the id of the code path.") },
    { "GetVerbosity", libxsmmpy_get_verbosity, METH_NOARGS,
      PyDoc_STR("Get the verbosity level.") },
    { "SetVerbosity", libxsmmpy_set_verbosity, METH_VARARGS,
      PyDoc_STR("Set the verbosity level.") },
    { NULL, NULL, 0, NULL } /* end of table */
  };
  PyObject *const pymod = Py_InitModule3("libxsmm", pymethod_def, PyDoc_STR(
    "Library targeting Intel Architecture for small, dense or "
    "sparse matrix multiplications, and small convolutions."));
  PyModule_AddIntConstant(pymod, "VERSION_API", LIBXSMM_VERSION2(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR));
  PyModule_AddIntConstant(pymod, "VERSION_ALL", LIBXSMM_VERSION4(LIBXSMM_VERSION_MAJOR, LIBXSMM_VERSION_MINOR,
                                                                LIBXSMM_VERSION_UPDATE, LIBXSMM_VERSION_PATCH));
  PyModule_AddIntConstant(pymod, "VERSION_MAJOR", LIBXSMM_VERSION_MAJOR);
  PyModule_AddIntConstant(pymod, "VERSION_MINOR", LIBXSMM_VERSION_MINOR);
  PyModule_AddIntConstant(pymod, "VERSION_UPDATE", LIBXSMM_VERSION_UPDATE);
  PyModule_AddIntConstant(pymod, "VERSION_PATCH", LIBXSMM_VERSION_PATCH);
  PyModule_AddStringConstant(pymod, "VERSION", LIBXSMM_VERSION);
  PyModule_AddStringConstant(pymod, "BRANCH", LIBXSMM_BRANCH);
  PyModule_AddIntConstant(pymod, "TARGET_ARCH_UNKNOWN", LIBXSMM_TARGET_ARCH_UNKNOWN);
  PyModule_AddIntConstant(pymod, "TARGET_ARCH_GENERIC", LIBXSMM_TARGET_ARCH_GENERIC);
  PyModule_AddIntConstant(pymod, "X86_IMCI", LIBXSMM_X86_IMCI);
  PyModule_AddIntConstant(pymod, "X86_GENERIC", LIBXSMM_X86_GENERIC);
  PyModule_AddIntConstant(pymod, "X86_SSE3", LIBXSMM_X86_SSE3);
  PyModule_AddIntConstant(pymod, "X86_SSE4", LIBXSMM_X86_SSE4);
  PyModule_AddIntConstant(pymod, "X86_AVX", LIBXSMM_X86_AVX);
  PyModule_AddIntConstant(pymod, "X86_AVX2", LIBXSMM_X86_AVX2);
  PyModule_AddIntConstant(pymod, "X86_AVX512", LIBXSMM_X86_AVX512);
  PyModule_AddIntConstant(pymod, "X86_AVX512_MIC", LIBXSMM_X86_AVX512_MIC);
  PyModule_AddIntConstant(pymod, "X86_AVX512_KNM", LIBXSMM_X86_AVX512_KNM);
  PyModule_AddIntConstant(pymod, "X86_AVX512_CORE", LIBXSMM_X86_AVX512_CORE);
  PyModule_AddIntConstant(pymod, "X86_AVX512_ICL", LIBXSMM_X86_AVX512_ICL);
  libxsmm_init(); /* initialize LIBXSMM */
}

#endif /*defined(__PYTHON) && defined(LIBXSMM_BUILD) && !defined(__STATIC)*/

