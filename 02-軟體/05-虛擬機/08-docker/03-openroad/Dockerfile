FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV HOME=/home/
ENV THE_PREFIX=/usr/
RUN apt update

# install basic packages
RUN apt install -y git wget vim cmake clang pkg-config curl zip unzip python3-dev 

# install 
RUN apt install verilator yosys

# install qemu riscv
RUN apt install -y opensbi qemu-system-misc u-boot-qemu qemu-user-static
RUN apt install -y gcc-riscv64-unknown-elf gcc-riscv64-linux-gnu

# install deno
RUN curl -fsSL https://deno.land/install.sh | sh

# COPY . .
COPY ./home /home/

# install openroad
RUN /home/install/install_openroad.sh

RUN source ./home/.bashrc

# ubuntu yosys repository is not compatible with other tools
# RUN apt install 
# RUN apt install -y yosys #
# RUN ./install_yosys.sh
