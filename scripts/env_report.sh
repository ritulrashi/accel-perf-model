#!/usr/bin/env bash
# Prints the provenance block that must accompany every reported result:
# time, commit, tree cleanliness, machine, and tool versions.
# Run it BEFORE writing any output file (docs/BUGS.md convention from
# rv32im-explorer bugs #2 and #7: an output file inside the tree makes it DIRTY).
set -uo pipefail
cd "$(dirname "$0")/.."

SC_PREFIX=${SYSTEMC_HOME:-/opt/systemc-3.0.2}

echo "date_utc: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "commit: $(git rev-parse HEAD 2>/dev/null || echo unknown)"
if ! git rev-parse --git-dir >/dev/null 2>&1; then
  echo "tree: NOT A GIT CHECKOUT (results not attributable to any commit)"
elif [ -n "$(git status --porcelain)" ]; then
  echo "tree: DIRTY (results not attributable to the commit above)"
else
  echo "tree: clean"
fi
echo "host: $(uname -srm)"
echo "os: $(. /etc/os-release && echo "$PRETTY_NAME")"
echo "lscpu:"
lscpu | grep -E '^(Architecture|Model name|CPU\(s\)|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|CPU max MHz|L1d|L1i|L2|L3)' | sed 's/^/  /'
echo "mem_total: $(grep MemTotal /proc/meminfo | awk '{print $2 " kB"}')"
echo "tools:"
echo "  c++: $(c++ --version | head -1)"
echo "  cmake: $(cmake --version | head -1)"
echo "  ninja: $(ninja --version)"
sc_ver=$(grep -hE '#define SC_VERSION_(MAJOR|MINOR|PATCH) ' "$SC_PREFIX/include/sysc/kernel/sc_ver.h" 2>/dev/null \
  | awk '{print $3}' | paste -sd. -)
# SC_VERSION_RELEASE_DATE is a two-line macro wrapping SYSTEMC_VERSION, so
# read the number from SYSTEMC_VERSION itself (docs/BUGS.md #2).
sc_date=$(grep -hE '#define SYSTEMC_VERSION ' "$SC_PREFIX/include/sysc/kernel/sc_ver.h" 2>/dev/null \
  | awk '{print $3}')
[ -n "${sc_date:-}" ] && sc_ver="$sc_ver (release date $sc_date)"
echo "  systemc: ${sc_ver:-not found} at $SC_PREFIX, commit $(cat "$SC_PREFIX/PINNED_COMMIT" 2>/dev/null || echo unknown)"
echo "  googletest: $(grep -oE 'GIT_TAG [0-9a-f]+ +# v[0-9.]+' CMakeLists.txt | sed 's/GIT_TAG //')"
echo "  python3: $(python3 --version 2>&1)"
echo "  numpy: $(python3 -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo not installed)"
echo "  verilator: $(verilator --version 2>/dev/null || echo not installed)"
