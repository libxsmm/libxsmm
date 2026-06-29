#!/usr/bin/env bash
set -euo pipefail

# Bump libxsmm version number and submit commit and tag automatically.
# Usage: scripts/release.sh <version>

tag=${1:?usage: $0 <tag>}
version=${tag#v}

case "${version}" in
  [0-9]*.[0-9]*)
    ;;
  *)
    echo "error: tag must be MAJOR.MINOR or MAJOR.MINOR.PATCH" >&2
    exit 1
    ;;
esac

IFS=. read -r major minor patch extra <<EOF
${version}
EOF

if [ -n "${extra:-}" ] || [ -z "${major:-}" ] || [ -z "${minor:-}" ]; then
  echo "error: invalid version: ${version}" >&2
  exit 1
fi

patch=${patch:-0}
version="${major}.${minor}.${patch}"

printf '%s\n' "${version}" > VERSION
git add VERSION

if ! git diff --cached --quiet; then
  git commit -m "Release: Bump version to ${version}"
fi

git tag "${tag}"
