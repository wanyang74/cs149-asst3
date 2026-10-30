#!/usr/bin/env bash
# One-shot setup for a fresh Lambda GPU instance (Ubuntu 24.04, x86_64).
# Usage: ./setup_remote.sh [--nsight]
#   --nsight  also install Nsight Compute (ncu) and Nsight Systems (nsys) for profiling
set -euo pipefail
trap 'echo "setup_remote.sh: failed at line $LINENO: $BASH_COMMAND" >&2' ERR
cd "$(dirname "$0")"

# Reference binaries (*_ref_x86) need glibc >= 2.38, i.e. Ubuntu 24.04+.
# (getconf avoids a `ldd | head` pipeline, which can SIGPIPE and trip pipefail.)
glibc=$(getconf GNU_LIBC_VERSION | awk '{print $2}')
echo "arch=$(uname -m) glibc=$glibc"
if [ "$(printf '%s\n' 2.38 "$glibc" | sort -V | head -1)" != "2.38" ]; then
    echo "ERROR: glibc $glibc < 2.38; reference binaries won't run. Relaunch with Ubuntu 24.04." >&2
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

sudo apt-get update -qq
sudo apt-get install -y -qq freeglut3-dev > /dev/null

if [ "${1:-}" = "--nsight" ]; then
    if ! dpkg -s cuda-keyring > /dev/null 2>&1; then
        wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb -O /tmp/cuda-keyring.deb
        sudo dpkg -i /tmp/cuda-keyring.deb > /dev/null
        sudo apt-get update -qq
    fi
    ncu_pkg=$(apt-cache search --names-only '^nsight-compute-[0-9]' | awk '{print $1}' | sort -V | tail -1)
    nsys_pkg=$(apt-cache search --names-only '^nsight-systems-[0-9]' | awk '{print $1}' | sort -V | tail -1)
    echo "installing $ncu_pkg $nsys_pkg"
    sudo apt-get install -y -qq "$ncu_pkg" "$nsys_pkg" > /dev/null
    ncu_dir=$(dirname "$(ls -d /opt/nvidia/nsight-compute/*/ncu | sort -V | tail -1)")
    nsys_dir=$(dirname "$(ls -d /opt/nvidia/nsight-systems/*/bin/nsys | sort -V | tail -1)")
    grep -q nsight ~/.bashrc || echo "export PATH=\"$ncu_dir:$nsys_dir:\$PATH\"" >> ~/.bashrc
    echo "Nsight installed; run 'source ~/.bashrc'. Use 'sudo \$(which ncu) ...' if you hit ERR_NVGPUCTRPERM."
fi

chmod +x scan/checker.py render/checker.py
for d in saxpy scan render; do
    make -s -C "$d" 2>&1 | grep -v "^$" || true
done
echo "SETUP_OK"
