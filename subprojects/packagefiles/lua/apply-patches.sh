#!/bin/sh
CURRENT_SOURCE_DIR="$1"
OUTDIR="$2"
# Remaining arguments are input files to be copied to the output after patching.
shift 2 >/dev/null 2>&1

cp -a "$@" "$OUTDIR"

OLDDIR="$(pwd)"
for p in "$CURRENT_SOURCE_DIR"/*.patch; do
    p="$(realpath $p)"
    cd "$OUTDIR"
    patch -p2 < "$p" >/dev/null 2>&1
    cd "$OLDDIR"
done
