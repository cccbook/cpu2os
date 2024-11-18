# 開放原始碼 EDA 工具

* 參考 -- https://pulp-platform.org/docs/coscup2024/phsauter_coscup24.pdf

1. Verilog: Verilog 轉 RTL ，並提供模擬功能
    * Icarus Verilog: 簡單易用 
    * Verilator: 可以與 SystemC / C++ 整合，速度快

2. Yosys: Synthesis tool
    * 將 RTL 轉為 Netlist

3. OpenRoad: Place & Route tool
    * 將 Netlist 轉為 GDS2

3. SystemVerilog: 
    * sv2v : Convert SystemVerilog to Verilog
    * svase: (pulp) SystemVerilog pre-elaborator
    * morty: (pulp) Morty reads SystemVerilog files and pickles them into a single file for easier handling. 

