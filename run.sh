#!/bin/bash
# Build Faiss with optimized AVX512.
# Exit immediately if any command fails
set -e

echo "╔═════════════════════════════════╗"
echo "║       Configuring Faiss ...     ║"
echo "╚═════════════════════════════════╝"
cmake -B build . \
    -DFAISS_OPT_LEVEL=avx512_spr \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=ON \
    -DBUILD_TESTING=OFF \
    -DCMAKE_BUILD_TYPE=Release \
    -DPython_VERSION=3.12 \
    -DPython_INCLUDE_DIRS="/usr/include/python3.12" \
    -DPython_NumPy_INCLUDE_DIRS="/usr/lib/python3/dist-packages/numpy/core/include"

# Build faiss
echo "╔══════════════════════════════════════════════════════╗"
echo "║           Building Faiss core library...             ║"
echo "╚══════════════════════════════════════════════════════╝"
make -C build -j faiss_avx512_spr

echo "╔══════════════════════════════════════════════════════╗"
echo "║         Faiss core library build complete!           ║"
echo "║         Building SWIG Python bindings...             ║"
echo "╚══════════════════════════════════════════════════════╝"
make -C build -j swigfaiss_avx512_spr

echo "╔══════════════════════════════════════════════════════╗"
echo "║        SWIG Python bindings build complete!          ║"
echo "║            Installing Python package...              ║"
echo "╚══════════════════════════════════════════════════════╝"
cd build/faiss/python
sudo python3 -m pip install . --break-system-packages
cd ../../..
echo "╔══════════════════════════════════════════════════════╗"
echo "║    All done! Faiss with Python bindings is ready!    ║"
echo "╚══════════════════════════════════════════════════════╝" 

