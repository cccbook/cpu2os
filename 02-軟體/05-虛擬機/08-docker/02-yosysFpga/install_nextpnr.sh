
NPROC=4

# install nextpnr

git clone --recursive https://github.com/YosysHQ/nextpnr.git
cd nextpnr
cmake . -DARCH=ecp5 -DTRELLIS_INSTALL_PREFIX=/usr/local
make -j$(nproc)
make install
exit 0
