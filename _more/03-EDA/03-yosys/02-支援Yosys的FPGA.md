

## ccc: 有哪些 FPGA 開發板支援 Yosys

ChatGPT: https://chatgpt.com/c/673b023a-1200-8012-b4e4-12ad218f8b7f

Yosys 是一個開源的邏輯綜合工具，支援多種 FPGA 平台，特別是那些具有開放工具鏈或能與 SymbiFlow 等開源工具鏈整合的 FPGA 開發板。以下是一些支援 Yosys 的主要 FPGA 開發板與相關平台：

---

### **1. Lattice FPGA**
Lattice 的 FPGA 是 Yosys 支援最好的平臺之一，因為 Lattice 提供了相對開放的設計文檔。
#### 支援的開發板：
- **ICEBreaker**  
  - 基於 Lattice iCE40 FPGA  
  - 專為開源硬體設計開發，與 Yosys 和 Project IceStorm 完美整合。
- **TinyFPGA 系列**  
  - 特別是 **TinyFPGA BX**（基於 Lattice iCE40 FPGA）。  
  - 適合個人項目和教育用途。
- **Upduino 系列**  
  - 小型、低成本的開發板，使用 iCE40 FPGA。
- **OrangeCrab**  
  - 基於 Lattice ECP5 FPGA，支援更高性能應用，能使用 Yosys 和 Trellis 工具鏈。

#### 相關工具：
- **Project IceStorm**：針對 Lattice iCE40 FPGA 的開源工具鏈。
- **Project Trellis**：針對 Lattice ECP5 FPGA 的開源工具鏈。

---

### **2. Xilinx FPGA**
雖然 Xilinx 主流 FPGA 使用專有工具鏈（如 Vivado），但部分開放的架構支援 Yosys 的工具鏈。
#### 支援的開發板：
- **Arty 系列（如 Arty A7）**  
  - 基於 Xilinx Artix-7 FPGA，部分功能可整合開源工具（如 SymbiFlow）。  
  - 適合學術和實驗用途。
- **Zynq 系列（如 ZedBoard）**  
  - 部分工具鏈可與 Yosys 整合，但多數情況下仍需依賴 Xilinx Vivado。

---

### **3. Gowin FPGA**
Gowin 是一家新興的 FPGA 製造商，近年來因為開放工具鏈的支援而受到關注。
#### 支援的開發板：
- **Tang Nano 系列**  
  - 如 **Tang Nano 9K** 和 **Tang Nano 4K**，是低成本、小型 FPGA 開發板，適合教育和簡單應用。  
  - 與 Yosys 的 Gowin 支援後端結合使用。

#### 相關工具：
- **APEX Toolchain**：針對 Gowin FPGA 的開源工具鏈。

---

### **4. Intel FPGA**
Intel FPGA（原 Altera）工具鏈主要是專有的，但 Yosys 提供部分 Cyclone 系列的初步支援。
#### 支援的開發板：
- **DE10-Nano**  
  - 使用 Cyclone V FPGA，部分綜合流程可嘗試 Yosys，但需要搭配 Quartus Prime 工具。

---

### **5. QuickLogic FPGA**
QuickLogic 是完全開放的 FPGA 製造商，與 Yosys 緊密合作。
#### 支援的開發板：
- **QuickFeather**  
  - 基於 QuickLogic EOS S3，支援完整的開源設計流程。

---

### **6. OpenFPGA 平台**
這些平臺專門針對開源 FPGA 設計，強調與 Yosys 整合。
#### 代表性開發板：
- **Fomu**  : https://github.com/im-tomu/fomu-hardware
  - 世界上最小的 FPGA 開發板，基於 Lattice iCE40UP，適合初學者。
- **Radiona ULX3S**  
  - 基於 Lattice ECP5 FPGA，適合高性能應用。

---

### 選擇指南
如果你是初學者或專注於開源工具，建議選擇 **Lattice iCE40 或 ECP5** 系列的 FPGA 開發板，如 ICEBreaker 或 OrangeCrab。  
對於更高性能應用，選擇 **Radiona ULX3S** 或 **Tang Nano 9K** 這類具備強大性能和開源工具鏈支援的開發板。