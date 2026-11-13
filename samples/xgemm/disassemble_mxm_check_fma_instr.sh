#!/usr/bin/env bash
# Disassemble all *.mxm files with xed64 and report compute instructions used inside
# (ACE top*, AMX tdp*, VNNI vpdp*/vpmadd*, VDP vdp*, Classic FMA vfmadd*/vfmsub*/vfnmadd*/vfnmsub*)

XED="/swtools/sde/latest/xed64"
if [ ! -x "${XED}" ]; then
  XED=$(command -v xed64 || command -v xed)
fi

if [ -z "${XED}" ]; then
  echo "Error: xed64 binary not found." >&2
  exit 1
fi

DIR="${1:-.}"

for file in "${DIR}"/*.mxm; do
  [ -e "${file}" ] || continue
  compute_inst=$("${XED}" -64 -ir "${file}" 2>/dev/null | grep -i -oE "\b(top|tdp|vpdp|vpmadd|vdp|vfmadd|vfmsub|vfnmadd|vfnmsub)[0-9a-zA-Z_]+" | sort -u | tr '\n' ' ')
  if [ -n "${compute_inst}" ]; then
    echo "${file}: ${compute_inst}"
  else
    echo "${file}: none"
  fi
done
