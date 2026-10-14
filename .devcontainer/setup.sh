#!/usr/bin/env bash
# Installs the full toolchain for all phases in one step. apt versions come
# from the Ubuntu 24.04 archive; SystemC is built from source at a pinned
# commit. scripts/env_report.sh records the exact versions next to every
# measured result. Also used by CI (Phase 6).
set -euo pipefail

SC_VERSION=3.0.2
SC_COMMIT=70b0fc8e4a74acc677b0fc73cea08f940c2115d5
SC_PREFIX=/opt/systemc-${SC_VERSION}

sudo apt-get update
sudo apt-get install -y --no-install-recommends \
  build-essential cmake ninja-build git ca-certificates \
  python3 python3-numpy python3-matplotlib \
  verilator

if [ "$(cat "${SC_PREFIX}/PINNED_COMMIT" 2>/dev/null || true)" != "${SC_COMMIT}" ]; then
  work=$(mktemp -d)
  git clone --quiet --depth 1 --branch "${SC_VERSION}" \
    https://github.com/accellera-official/systemc "${work}/systemc"
  got=$(git -C "${work}/systemc" rev-parse HEAD)
  if [ "${got}" != "${SC_COMMIT}" ]; then
    echo "SystemC tag ${SC_VERSION} is ${got}, expected pinned ${SC_COMMIT}" >&2
    exit 1
  fi
  # C++17 must match the project: SystemC embeds the language version in a
  # link-time API check symbol.
  cmake -S "${work}/systemc" -B "${work}/build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_STANDARD=17 \
    -DCMAKE_INSTALL_PREFIX="${SC_PREFIX}" -DBUILD_SHARED_LIBS=ON
  cmake --build "${work}/build"
  sudo cmake --install "${work}/build"
  echo "${SC_COMMIT}" | sudo tee "${SC_PREFIX}/PINNED_COMMIT" >/dev/null
  rm -rf "${work}"
fi
echo "SystemC ${SC_VERSION} (${SC_COMMIT}) installed in ${SC_PREFIX}"
