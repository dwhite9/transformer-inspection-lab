#!/usr/bin/env bash
# Rebuild both HTML guides from their markdown sources.
# Each build script declares its own dependencies (PEP 723), so `uv run` needs no setup.

# "Strict mode": stop on the first failed command (-e), on use of an unset variable (-u),
# and when any part of a pipeline fails (-o pipefail).
# Same as writing:
#   set -e
#   set -u
#   set -o pipefail
set -euo pipefail

# The folder this script lives in, so it works from any directory.
#   ${BASH_SOURCE[0]}   the path this script was run as
#   dirname             strips the file name, leaving the folder
#   cd ... && pwd       turns that into an absolute path
# Long form:
#   script_path="${BASH_SOURCE[0]}"
#   script_dir="$(dirname "$script_path")"
#   cd "$script_dir"
#   here="$(pwd)"
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

uv run "$here/build_manual.py"
uv run "$here/build_study_guide.py"
