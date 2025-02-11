
## ccc: 請給一個 symbiFlow 的輸入輸出使用範例

ChatGPT: https://chatgpt.com/c/673b0564-2324-8012-820e-0b4df855fa6c

以下是一個使用 SymbiFlow 工具鏈的簡單範例，從 RTL 描述到生成位流，適用於 Lattice FPGA，例如 Lattice iCE40 系列。

### 前置準備
1. 安裝 SymbiFlow 工具鏈。
   - 可參考 [SymbiFlow 安裝指南](https://symbiflow.github.io/)。
   - 確保已安裝以下工具：
     - Yosys（合成工具）
     - nextpnr（佈局布線工具）
     - IceStorm（生成位流工具，針對 iCE40 FPGA）

2. 設備準備：使用一塊 Lattice iCE40 FPGA 開發板（如 iCEBreaker）。

---

### 範例流程：將一個簡單的 LED 閃爍程式映射到 FPGA

#### 1. RTL 描述（`blink.v`）
設計一個簡單的 LED 閃爍模塊：

```verilog
module blink (
    input wire clk,
    output wire led
);
    reg [23:0] counter = 0;
    always @(posedge clk) begin
        counter <= counter + 1;
    end
    assign led = counter[23]; // 每秒閃爍一次 (假設 clk = 12MHz)
endmodule
```

#### 2. 合成設計
使用 **Yosys** 將 Verilog 源碼合成為中間表示（`blink.json`）：

```bash
yosys -p "synth_ice40 -json blink.json" blink.v
```

- **輸入**：
  - `blink.v`（Verilog 源碼）
- **輸出**：
  - `blink.json`（中間表示文件）

---

#### 3. 佈局和布線
使用 **nextpnr** 完成佈局和布線，目標為 iCE40 FPGA：

```bash
nextpnr-ice40 --hx8k --package ct256 --json blink.json --asc blink.asc
```

- **參數解釋**：
  - `--hx8k`：目標設備為 iCE40 HX8K。
  - `--package ct256`：設備封裝類型。
  - `--json blink.json`：合成輸出的中間表示文件。
  - `--asc blink.asc`：生成的 FPGA 佈局布線文件。

- **輸入**：
  - `blink.json`
- **輸出**：
  - `blink.asc`（FPGA 配置文件）

---

#### 4. 生成位流
使用 **IceStorm** 工具將佈局布線文件轉換為 FPGA 可燒錄的位流文件：

```bash
icepack blink.asc blink.bin
```

- **輸入**：
  - `blink.asc`
- **輸出**：
  - `blink.bin`（位流文件）

---

#### 5. 將位流燒錄到 FPGA
使用 **IceStorm** 提供的工具（如 `iceprog`）燒錄位流到 FPGA：

```bash
iceprog blink.bin
```

- **輸入**：
  - `blink.bin`
- **輸出**：
  - FPGA 配置完成，LED 開始閃爍。

---

### 輸入與輸出總結

#### **輸入**：
1. RTL 描述（`blink.v`）
2. 工具參數（目標設備、封裝等）

#### **輸出**：
1. 中間表示（`blink.json`）
2. 佈局布線文件（`blink.asc`）
3. 位流文件（`blink.bin`）
4. 燒錄成功的 FPGA 設計（LED 閃爍）

---

以上是 SymbiFlow 的基本使用流程範例，完整的流程可根據具體設備和需求進行調整。