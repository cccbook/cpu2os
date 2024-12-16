
NPROC=4

# install prjtrellis
git clone --recursive https://github.com/YosysHQ/prjtrellis
cd prjtrellis/libtrellis
cmake -DCMAKE_INSTALL_PREFIX=/usr/local .
make -j${NPROC}
make install
exit 0
