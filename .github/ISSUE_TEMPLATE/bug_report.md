---
name: Bug report
about: Create a report to help us improve.

---

Please help with a clear and concise description of what the bug is, and what you expected to happen. Feel free to paste or attach a reproducer or some sample code, but at least mention how to build LIBXSMM as well as the kind of system exposing the issue (CPU architecture, operating system, etc.).

```bash
make
```

Determine accurate build information in case of an upstreamed package (binary distribution):

```bash
export LIBXSMM_DUMP_BUILD=1
./application_linked_with_libxsmm ...
```

Build information is available when LIBXSMM was dynamically or statically linked with an application (not for header-only).

