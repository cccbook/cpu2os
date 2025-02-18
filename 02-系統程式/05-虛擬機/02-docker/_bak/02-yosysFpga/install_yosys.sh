
NPROC=4

# install yosys
git clone https://github.com/YosysHQ/yosys yosys
cd yosys
make -j$(nproc)
# make test #optional, required iverilog
make install
exit 0