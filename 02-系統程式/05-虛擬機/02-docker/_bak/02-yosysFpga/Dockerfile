FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/workspace/
ENV THE_PREFIX=/usr/
RUN apt update

# install deno
RUN curl -fsSL https://deno.land/install.sh | sh

# install yosysHQ toolkit for FPGA
RUN apt install -y git wget
RUN apt install -y build-essential clang tcl-dev libreadline-dev pkg-config bison flex \
  iverilog gtkwave libftdi-dev cmake python3-dev libboost-all-dev libeigen3-dev
COPY . .
# ubuntu yosys repository is not compatible with other tools
# RUN apt install -y yosys #
RUN ./install_yosys.sh
RUN ./install_prjtrellis.sh
RUN ./install_nextpnr.sh
RUN ./install_systemc.sh
RUN ./install_verilator.sh