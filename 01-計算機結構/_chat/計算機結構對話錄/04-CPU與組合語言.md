# CPU與組合語言

* [AI對話](https://aistudio.google.com/app/prompts?state=%7B%22ids%22:%5B%221RddlBt3Q3Y4SEEYzQlQQHj1M98nCrG2K%22%5D,%22action%22:%22open%22,%22userId%22:%22111605452542833299008%22,%22resourceKeys%22:%7B%7D%7D&usp=sharing)

## Q: 請設計一個 CPU ，先從設計其組合語言開始

好的，同學請坐！

歡迎來到「數位邏輯與計算機結構」這門課。我是你的老師，接下來的一年，我會帶你一步步地探索電腦的內心世界。這門課聽起來很硬，但請不要擔心，我會用最白話的方式，帶你從無到有，親手打造一顆屬於你自己的 CPU。

我們的旅程，不是從畫電路圖開始，也不是從寫程式開始，而是從**「創造一種語言」**開始。

想像一下，你要訓練一隻非常聽話、但只懂 0 和 1 的超級小狗。你必須先設計一套牠聽得懂的「指令」，比如「坐下」、「撿東西」、「叫一聲」。這套指令就是這隻小狗的「組合語言」。

CPU 就是這隻小狗，而我們今天要做的，就是為我們的 CPU 設計一套專屬的「組合語言」(Assembly Language)，也稱為**指令集架構 (Instruction Set Architecture, ISA)**。ISA 是軟體與硬體之間的橋樑，是軟體工程師與硬體工程師的「契約」。

---

### 第一步：思考 CPU 的基本能力

我們的 CPU 需要有哪些基本能力？就像我們思考小狗需要會做什麼一樣。

1.  **記憶能力**：牠需要有地方暫時存放東西。就像小狗的嘴巴可以叼著球。在 CPU 裡，這些超高速的暫存空間稱為**暫存器 (Registers)**。
2.  **算術能力**：牠需要會加、減、乘、除等基本數學。這是 CPU 的核心功能。
3.  **邏輯能力**：牠需要能做「而且 (AND)」、「或者 (OR)」、「不是 (NOT)」這樣的邏輯判斷。
4.  **搬運能力**：牠需要能把東西從一個地方搬到另一個地方。例如，從「記憶體 (RAM)」這個大倉庫，搬到「暫存器」這個工作檯上。
5.  **決策能力**：牠需要能根據情況做出判斷，決定下一步要做什麼。例如，「如果嘴裡的球是紅色的，就往左跳；否則，就往右跳」。這就是所謂的**分支 (Branch)** 和**跳躍 (Jump)**。

基於這些想法，我們來設計一個極簡的 CPU，就叫它 **`SimpleCPU-1`** 吧！

### 第二步：設計 `SimpleCPU-1` 的規格

#### 1. 暫存器 (Registers)

我們給 `SimpleCPU-1` 配備 8 個 32 位元 (bit) 的通用暫存器，名字很簡單，就叫 `$R0`, `$R1`, `$R2`, ..., `$R7`。

*   `$R0` 比較特別，我們規定它永遠是 0。寫入任何值給它，它仍然是 0。這在硬體設計上有很多好處。
*   其他的 `$R1` ~ `$R7` 都可以自由使用。

另外，還需要一個特殊的暫存器：

*   **程式計數器 (Program Counter, PC)**：它存放著「下一條要執行的指令」所在的記憶體位址。CPU 每執行完一條指令，PC 就會自動指向下一條。

#### 2. 記憶體 (Memory)

想像成一個有著很多門牌號碼的巨大櫃子，每個櫃子可以存放一個 32 位元的資料。CPU 可以透過位址 (address) 來讀取或寫入資料。

#### 3. 指令格式

我們採用**精簡指令集 (RISC)** 的精神：

*   **指令長度固定**：所有指令都是 32 位元長。這樣硬體處理起來比較簡單。
*   **格式規律**：指令的格式就那幾種，方便硬體解析。

我們設計兩種主要的指令格式：

**R-Type (Register-Type)：用於暫存器之間的操作**

```
| opcode (6) | rs1 (3) | rs2 (3) | rd (3) | funct (17) |
```

*   `opcode`: 指令類型碼 (例如：算術指令)
*   `rs1`: 來源暫存器 1 (Source Register 1)
*   `rs2`: 來源暫存器 2 (Source Register 2)
*   `rd`:  目的暫存器 (Destination Register)
*   `funct`: 功能碼 (例如：ADD, SUB)

**I-Type (Immediate-Type)：用於暫存器和一個立即數 (常數) 之間的操作**

```
| opcode (6) | rs1 (3) | rd (3) | immediate (20) |
```

*   `opcode`: 指令類型碼 (例如：載入、儲存、立即數算術)
*   `rs1`: 來源暫存器 1
*   `rd`:  目的暫存器
*   `immediate`: 一個直接寫在指令裡的常數值

---

### 第三步：定義 `SimpleCPU-1` 的組合語言指令集

好，契約來了！這就是我們 `SimpleCPU-1` 的說明書。

#### 1. 算術/邏輯指令 (R-Type)

*   `ADD rd, rs1, rs2`
    *   **說明**: `rd = rs1 + rs2`
    *   **範例**: `ADD $R1, $R2, $R3`  (把 `$R2` 和 `$R3` 的值相加，結果存到 `$R1`)

*   `SUB rd, rs1, rs2`
    *   **說明**: `rd = rs1 - rs2`
    *   **範例**: `SUB $R4, $R5, $R6`

*   `AND rd, rs1, rs2`
    *   **說明**: `rd = rs1 & rs2` (位元 AND)
    *   **範例**: `AND $R1, $R2, $R3`

*   `OR rd, rs1, rs2`
    *   **說明**: `rd = rs1 | rs2` (位元 OR)
    *   **範例**: `OR $R1, $R2, $R3`

#### 2. 立即數指令 (I-Type)

*   `ADDI rd, rs1, immediate`
    *   **說明**: `rd = rs1 + immediate` (立即數加法)
    *   **範例**: `ADDI $R1, $R2, 100` (把 `$R2` 的值加上 100，結果存到 `$R1`)

#### 3. 資料搬移指令 (I-Type)

CPU 的計算都在暫存器裡，所以我們需要指令把資料在記憶體和暫存器之間搬來搬去。這叫做**載入/儲存架構 (Load/Store Architecture)**。

*   `LW rd, immediate(rs1)` (Load Word)
    *   **說明**: 從記憶體讀取資料到暫存器。記憶體位址由 `rs1 + immediate` 算出。
    *   **範例**: `LW $R1, 64($R2)` (計算 `$R2` 的值 + 64 得到記憶體位址，然後從該位址讀取資料，存到 `$R1`)

*   `SW rd, immediate(rs1)` (Store Word)
    *   **說明**: 將暫存器的資料寫入記憶體。記憶體位址由 `rs1 + immediate` 算出。
    *   **範例**: `SW $R1, 64($R2)` (計算 `$R2` 的值 + 64 得到記憶體位址，然後把 `$R1` 的資料寫入該位址)

#### 4. 分支指令 (I-Type)

讓程式可以跳躍執行，這是實現 `if-else`、`for` 迴圈的基礎。

*   `BEQ rs1, rd, immediate` (Branch if Equal)
    *   **說明**: 如果 `rs1` 和 `rd` 的值相等，則跳轉到 `PC + immediate` 的位址去執行。
    *   **範例**: `BEQ $R1, $R2, 8` (如果 `$R1` 和 `$R2` 相等，則跳過下面兩條指令，跳到 `PC+8` 的地方繼續)

---

### 第四步：寫第一個 `SimpleCPU-1` 程式

讓我們來用剛剛定義的語言，寫一個簡單的程式：計算 `10 + 20`，並將結果存到記憶體位址 `100`。

```assembly
# 程式目標: 計算 10 + 20，結果存到記憶體位址 100

# 步驟 1: 把 10 放到暫存器 $R1
# 我們利用 $R0 (永遠是 0) 來達成
# $R1 = $R0 + 10
ADDI $R1, $R0, 10

# 步驟 2: 把 20 放到暫存器 $R2
# $R2 = $R0 + 20
ADDI $R2, $R0, 20

# 步驟 3: 計算 $R1 + $R2，結果存到 $R3
# $R3 = $R1 + $R2  (此時 $R3 的值會是 30)
ADD $R3, $R1, $R2

# 步驟 4: 把結果 30 存到記憶體位址 100
# 我們用 $R0 + 100 來定位記憶體位址
# Memory[100] = $R3
SW $R3, 100($R0)

# 程式結束
# (真實世界中會有一個 HALT 指令來停止 CPU)
```

看到這裡，你可能會想：「這比 Python 或 C++ 麻煩太多了吧！」

是的，這就是組合語言。它非常囉嗦，但每一行都精確地對應到 CPU 的一個具體動作。作為一個 CS 學生，理解這一層是打通任督二脈的關鍵。

### 接下來呢？從語言到硬體

今天，我們定義了 `SimpleCPU-1` 的「靈魂」——它的指令集。我們知道它能做什麼、怎麼跟它溝通。

我們的下一個任務，就是用**數位邏輯電路**來打造 `SimpleCPU-1` 的「肉體」。我們會學到：

1.  **算術邏輯單元 (ALU)**：如何用邏輯閘 (AND, OR, NOT) 做出一個可以執行 `ADD`, `SUB`, `AND`, `OR` 的計算機。
2.  **暫存器檔案 (Register File)**：如何設計一個可以快速讀寫 `$R0` ~ `$R7` 的電路。
3.  **控制單元 (Control Unit)**：CPU 的大腦！它會讀取指令的 `opcode`，然後像個交通警察一樣，指揮 ALU、暫存器、記憶體在正確的時間做正確的事。

我們將會使用一種叫做**硬體描述語言 (Hardware Description Language, HDL)** 的工具，其中最流行的就是 **Verilog**。它長得有點像 C 語言，但我們不是在寫軟體，而是在「描述硬體」。

舉個例子，一個簡單的 32 位元加法器，用 Verilog 描述起來會是這樣：

```verilog
// 這是一個描述 32 位元加法器的 Verilog 模組
module Adder32 (
    input  [31:0] a,     // 輸入 a
    input  [31:0] b,     // 輸入 b
    output [31:0] sum    // 輸出 sum
);

    // 把 a 和 b 相加，結果指定給 sum
    assign sum = a + b;

endmodule
```

是不是很有趣？我們用程式碼來「定義」一個硬體元件！

**總結一下今天的課程：**

我們從零開始，為一個虛構的 `SimpleCPU-1` 設計了它的核心——**指令集架構 (ISA)**。這套指令集就是我們與這顆 CPU 溝通的語言。我們定義了暫存器、指令格式、以及四種類型的基本指令。最後，我們還用這個新語言寫了第一支程式。

這份「契約」已經確立。接下來，我們就要以這份契約為藍圖，開始我們的硬體建造之旅了！

有任何問題，隨時可以提出來！

## Q: 好，那現在開始用 Verilog 設計出該 CPU

很好！同學，你的求知慾非常旺盛，這正是成為一位優秀工程師的特質。

我們已經有了 `SimpleCPU-1` 的藍圖——指令集架構(ISA)。現在，我們要戴上硬體工程師的帽子，把這個藍圖變成實際的電路。我們將使用 Verilog 這個「硬體界的 C 語言」來描述我們的設計。

想像一下，我們要蓋一棟房子 (CPU)。我們不會一開始就去砌每一塊磚 (邏輯閘)。相反地，我們會先設計好各個房間 (功能單元)，比如客廳 (ALU)、臥室 (暫存器)、廚房 (記憶體)，然後再把它們組裝起來。

### 宏觀架構：CPU 的兩大核心

我們的 `SimpleCPU-1`，像所有現代 CPU 一樣，可以被切分成兩個主要部分：

1.  **資料路徑 (Datapath)**：這是 CPU 的「肌肉和骨骼」。它包含了所有處理和儲存資料的硬體，例如：
    *   程式計數器 (PC)
    *   算術邏輯單元 (ALU)
    *   暫存器檔案 (Register File)
    *   記憶體介面
2.  **控制單元 (Control Unit)**：這是 CPU 的「大腦」。它會解析指令，然後像個交響樂指揮家一樣，發送信號給資料路徑的各個部分，告訴它們這一瞬間該做什麼事 (例如：現在要讀暫存器、現在要做加法、現在要把結果寫回暫存器)。

我們的策略就是：**先建造資料路徑的各個元件，然後設計控制單元，最後把它們完美地組裝起來！**

---

### 第一步：定義指令的二進位格式

在開始寫 Verilog 之前，我們要先把上一堂課的「組合語言」翻譯成 CPU 真正看得懂的「機器語言」(0 和 1)。

我們來定義指令中的 `opcode`：

| 指令類型 | Opcode (6 bits) |
| :--- | :--- |
| R-Type (ADD, SUB...) | `6'b000000` |
| ADDI | `6'b001000` |
| LW | `6'b100011` |
| SW | `6'b101011` |
| BEQ | `6'b000100` |

對於 R-Type 指令，我們還需要用 `funct` 欄位來區分是 ADD 還是 SUB：

| 指令 | Funct |
| :--- | :--- |
| ADD | `17'h00020` |
| SUB | `17'h00022` |

為了方便管理，我們通常會把這些定義放在一個檔案裡，例如 `defines.v`：

```verilog
// defines.v
// 指令 Opcodes
`define OPCODE_RTYPE   6'b000000
`define OPCODE_ADDI    6'b001000
`define OPCODE_LW      6'b100011
`define OPCODE_SW      6'b101011
`define OPCODE_BEQ     6'b000100

// R-Type 的 Funct
`define FUNCT_ADD      17'h00020
`define FUNCT_SUB      17'h00022
```

---

### 第二步：建造資料路徑 (Datapath) 的元件

#### 1. 程式計數器 (Program Counter, PC)

PC 的工作很單純：在每個時脈週期 (clock cycle)，將自己的值更新成下一個指令的位址。

```verilog
// pc.v
module ProgramCounter (
    input             clk,         // 時脈訊號
    input             rst,         // 重置訊號
    input      [31:0] pc_in,       // 下一個 PC 的值
    output reg [31:0] pc_out       // 目前 PC 的值
);

    always @(posedge clk or posedge rst) begin
        if (rst) begin
            pc_out <= 32'b0; // 電腦啟動時，從位址 0 開始
        end else begin
            pc_out <= pc_in; // 在每個時脈上升緣，更新 PC
        end
    end

endmodule
```
*   `always @(posedge clk)` 是 Verilog 的核心語法，代表「每當時脈訊號從 0 變成 1 的那一瞬間，就執行裡面的程式碼」。這就是數位電路中「同步」的概念。

#### 2. 算術邏輯單元 (Arithmetic Logic Unit, ALU)

ALU 是 CPU 的計算核心。它接收兩個運算元和一個控制信號，然後輸出計算結果。

```verilog
// alu.v
module ALU (
    input      [31:0] a,         // 運算元 A
    input      [31:0] b,         // 運算元 B
    input      [3:0]  alu_op,    // 控制信號，決定要做哪種運算
    output reg [31:0] result,    // 運算結果
    output reg        zero       // 結果是否為 0 的旗標
);
    // 為了方便閱讀，我們先定義各種運算對應的 alu_op
    localparam ALU_ADD = 4'b0010;
    localparam ALU_SUB = 4'b0110;
    localparam ALU_AND = 4'b0000;
    localparam ALU_OR  = 4'b0001;
    // ... 可以再加入更多運算

    always @(*) begin // 組合邏輯，輸入一變，輸出就馬上跟著變
        case (alu_op)
            ALU_ADD: result = a + b;
            ALU_SUB: result = a - b;
            ALU_AND: result = a & b;
            ALU_OR:  result = a | b;
            default: result = 32'b0; // 預設輸出
        endcase

        if (result == 32'b0) begin
            zero = 1'b1;
        end else begin
            zero = 1'b0;
        end
    end
endmodule
```
*   `BEQ` 指令會需要知道兩個數相減的結果是否為 0，所以我們多設計了一個 `zero` 輸出。

#### 3. 暫存器檔案 (Register File)

這就像是 CPU 的工作檯，有 8 個抽屜 (`$R0`~`$R7`)。它需要能同時讀取兩個暫存器的值 (給 `rs1`, `rs2`)，並根據指令要求，將一個值寫入指定的暫存器 (`rd`)。

```verilog
// reg_file.v
module RegisterFile (
    input         clk,
    input         rst,
    input         reg_write_en,      // 致能信號，1 代表要寫入
    input  [2:0]  read_addr1,      // rs1 的位址
    input  [2:0]  read_addr2,      // rs2 的位址
    input  [2:0]  write_addr,      // rd 的位址
    input  [31:0] write_data,      // 要寫入的資料
    output [31:0] read_data1,      // rs1 讀出的資料
    output [31:0] read_data2       // rs2 讀出的資料
);
    // 宣告一個記憶體陣列來模擬 8 個 32-bit 的暫存器
    reg [31:0] registers [0:7];

    // 寫入操作 (同步)
    always @(posedge clk) begin
        // 只有在 reg_write_en 為 1 且目標不為 $R0 時才寫入
        if (reg_write_en && write_addr != 3'b0) begin
            registers[write_addr] <= write_data;
        end
    end

    // 讀取操作 (非同步)
    // $R0 永遠是 0
    assign read_data1 = (read_addr1 == 3'b0) ? 32'b0 : registers[read_addr1];
    assign read_data2 = (read_addr2 == 3'b0) ? 32'b0 : registers[read_addr2];

    // 初始化 (for simulation)
    integer i;
    initial begin
      for (i=0; i<8; i=i+1)
        registers[i] = 0;
    end

endmodule
```
*   **讀取**是**非同步 (asynchronous)**的：只要地址一給定，資料就立刻出現在輸出端。
*   **寫入**是**同步 (synchronous)**的：只有在時脈上升緣且寫入致能 (`reg_write_en`) 為 1 時，資料才會被寫入。這是為了確保資料在穩定的時候才被寫入，避免混亂。

---

### 第三步：設計大腦 (Control Unit)

控制單元是設計中最精華的部分。它不處理資料，只負責解碼和發號施令。

它的輸入是 32 位元指令中代表操作類型 `opcode` 欄位 (`instr[31:26]`)。
它的輸出是一系列的**控制信號**，告訴資料路徑各元件該如何動作。

```verilog
// control_unit.v
`include "defines.v"

module ControlUnit (
    input  [5:0] opcode,
    output reg   reg_write_en, // 是否要寫入暫存器?
    output reg   mem_to_reg,   // 寫回暫存器的資料來自記憶體(LW)還是ALU?
    output reg   mem_read,     // 是否要讀取記憶體?
    output reg   mem_write,    // 是否要寫入記憶體?
    output reg [3:0] alu_op,   // 要請 ALU 做什麼運算?
    output reg   alu_src,      // ALU 的第二個運算元來自暫存器還是立即數?
    output reg   branch        // 是否是 BEQ 指令?
);
    // 為了 ALU 控制，定義運算
    localparam ALU_ADD = 4'b0010;
    localparam ALU_SUB = 4'b0110;

    always @(*) begin
        // 先設定一組安全的預設值
        reg_write_en = 1'b0; mem_to_reg   = 1'b0; mem_read     = 1'b0;
        mem_write    = 1'b0; alu_src      = 1'b0; branch       = 1'b0;
        alu_op       = 4'b0; // 預設不做事

        case (opcode)
            `OPCODE_RTYPE: begin
                reg_write_en = 1'b1; // ADD/SUB 要寫回暫存器
                alu_src      = 1'b0; // 運算元 2 來自暫存器
                alu_op       = ALU_SUB; // 暫定 SUB，之後會根據 funct 欄位細分
            end
            `OPCODE_ADDI: begin
                reg_write_en = 1'b1; // 要寫回暫存器
                alu_src      = 1'b1; // 運算元 2 來自立即數
                alu_op       = ALU_ADD;
            end
            `OPCODE_LW: begin
                reg_write_en = 1'b1; // 要寫回暫存器
                mem_to_reg   = 1'b1; // 寫回的資料來自記憶體
                mem_read     = 1'b1; // 要讀記憶體
                alu_src      = 1'b1; // 位址計算用到立即數
                alu_op       = ALU_ADD; // 位址計算是 base + offset
            end
            `OPCODE_SW: begin
                mem_write    = 1'b1; // 要寫記憶體
                alu_src      = 1'b1; // 位址計算用到立即數
                alu_op       = ALU_ADD; // 位址計算是 base + offset
            end
            `OPCODE_BEQ: begin
                branch       = 1'b1; // 是分支指令
                alu_src      = 1'b0; // 比較兩個暫存器
                alu_op       = ALU_SUB; // 用相減來判斷是否相等
            end
            default: begin
                // ... handle illegal instruction
            end
        endcase
    end
endmodule
```

---

### 第四步：組裝！`SimpleCPU-1` 頂層模組

終於，我們可以把所有零件組裝起來了。這就像在主機板上插上 CPU、記憶體、顯示卡一樣。

```verilog
// SimpleCPU.v
`include "defines.v"

module SimpleCPU (
    input clk,
    input rst
);

    // --- 連接線 (Wires) ---
    wire [31:0] pc_current, pc_next, pc_plus_4;
    wire [31:0] instruction;
    wire [31:0] reg_read_data1, reg_read_data2;
    wire [31:0] alu_input_b;
    wire [31:0] immediate_extended;
    wire [31:0] alu_result;
    wire        alu_zero;
    wire [31:0] mem_read_data;
    wire [31:0] reg_write_data;
    wire        branch_taken;

    // --- 控制信號線 ---
    wire        reg_write_en, mem_to_reg, mem_read, mem_write, alu_src, branch;
    wire [3:0]  alu_op_ctrl;

    // --- 1. PC 和指令抓取 (Instruction Fetch) ---
    ProgramCounter pc_unit (
        .clk(clk), .rst(rst),
        .pc_in(pc_next),
        .pc_out(pc_current)
    );
    
    assign pc_plus_4 = pc_current + 4; // PC 預設指向下一條指令

    InstructionMemory instr_mem ( // 假設我們有一個指令記憶體模組
        .address(pc_current),
        .data_out(instruction)
    );

    // --- 2. 指令解碼與讀取暫存器 (Decode & Register Read) ---
    ControlUnit ctrl_unit (
        .opcode(instruction[31:26]),
        .reg_write_en(reg_write_en), .mem_to_reg(mem_to_reg), .mem_read(mem_read),
        .mem_write(mem_write), .alu_op(alu_op_ctrl), .alu_src(alu_src), .branch(branch)
    );

    RegisterFile reg_file (
        .clk(clk), .rst(rst),
        .reg_write_en(reg_write_en),
        .read_addr1(instruction[25:23]), // rs1
        .read_addr2(instruction[22:20]), // rs2
        .write_addr(instruction[19:17]), // rd
        .write_data(reg_write_data),
        .read_data1(reg_read_data1),
        .read_data2(reg_read_data2)
    );
    
    // --- 立即數處理 ---
    // 這邊簡化了，只處理 I-Type 的 immediate
    assign immediate_extended = {{12{instruction[19]}}, instruction[19:0]}; // 符號擴展

    // --- 3. 執行 (Execute) ---
    // 多工器 (MUX) 決定 ALU 的第二個輸入
    assign alu_input_b = (alu_src == 1'b0) ? reg_read_data2 : immediate_extended;
    
    ALU alu_unit (
        .a(reg_read_data1),
        .b(alu_input_b),
        .alu_op(alu_op_ctrl),
        .result(alu_result),
        .zero(alu_zero)
    );

    // --- 4. 記憶體存取 (Memory Access) ---
    DataMemory data_mem ( // 假設我們有一個資料記憶體模組
        .clk(clk),
        .address(alu_result), // LW/SW 的位址由 ALU 計算
        .write_data(reg_read_data2),
        .mem_write(mem_write),
        .mem_read(mem_read),
        .read_data(mem_read_data)
    );

    // --- 5. 寫回 (Write Back) ---
    // 多工器 (MUX) 決定要寫回暫存器的資料來源
    assign reg_write_data = (mem_to_reg == 1'b1) ? mem_read_data : alu_result;

    // --- PC 更新邏輯 ---
    assign branch_taken = branch & alu_zero; // BEQ 且 ALU 結果為 0
    // 分支位址 = PC+4 + (offset << 2) -> 這裡簡化成直接用 immediate
    assign pc_next = (branch_taken) ? (pc_plus_4 + immediate_extended) : pc_plus_4;

endmodule
```

### 總結與下一步

恭喜你！我們剛剛用紙和 Verilog，從邏輯閘的層次，一步步地搭建出了一顆能運作的 CPU 的核心架構。

*   我們設計了 `PC`, `ALU`, `Register File` 等**資料路徑**元件。
*   我們設計了 `Control Unit` 這個**大腦**來指揮它們。
*   我們在 `SimpleCPU` 這個頂層模組中，用 `wire` 把所有元件**連接**起來，讓資料和控制信號能夠正確地流動。

你可能已經發現，CPU 的運作就像一條精密的**生產線**：
1.  **取指令 (Fetch)**：PC 指向記憶體，拿出指令。
2.  **解碼 (Decode)**：控制單元看懂指令，並從暫存器拿出需要的資料。
3.  **執行 (Execute)**：ALU 進行計算。
4.  **記憶體 (Memory)**：需要的話，就讀寫記憶體。
5.  **寫回 (Write Back)**：把最終結果寫回暫存器。

這五個步驟，就是鼎鼎大名的**「五階段管線 (Five-Stage Pipeline)」**的基礎。我們今天設計的是一個比較簡單的**「單週期 (Single-Cycle)」CPU**，也就是一條指令花一個時脈週期完成。下一步，我們將會探討如何透過管線化 (Pipelining) 技術，讓 CPU 在同一個時間內處理多條指令，大幅提升效率！

現在，你可以把這些 Verilog 程式碼放到模擬器 (Simulator) 裡，載入我們上一堂課寫的 `10 + 20` 的機器碼，然後觀察每個時脈週期，PC、暫存器、記憶體裡的數值變化。當你看到 `$R3` 的值變成 30，然後記憶體位址 100 的地方也被寫入 30 時，那種「點亮」的成就感，就是電腦科學最迷人的地方！

## Q: 請先不要使用管線機制，重新設計一個簡單的 Single Cycle CPU

好的！同學，問得非常好！

你察覺到了一個關鍵點。在我之前的介紹中，為了展示 CPU 的所有功能階段，我直接跳到了最終的組裝圖。但要真正理解其運作原理，我們必須從最簡單、最直觀的模型開始。這就是**單週期 CPU (Single-Cycle CPU)**。

在你蓋摩天大樓（管線化 CPU）之前，必須先學會如何蓋一棟堅固的小木屋（單週期 CPU）。這個基礎打好了，之後的一切都會水到渠成。

---

### 單週期 CPU 的核心哲學

想像一下一間手作麵包店。單週期 CPU 的運作方式就像是：

> 一位麵包師傅，從頭到尾**獨自完成一整個麵包**（揉麵、發酵、烘烤、包裝），然後才開始做下一個。

這個過程的特點是：
*   **簡單直觀**：一個任務從頭做到尾，不會搞混。
*   **耗時**：一個時脈週期 (Clock Cycle) 的長度，必須由**最耗時**的那個指令決定（例如，需要存取記憶體的 `LW` 指令，就像做最複雜的千層麵包一樣）。即使是簡單的 `ADD` 指令（就像烤個小餅乾），也必須等待同樣長的時脈時間。

這就是「單週期」的意義：**一條指令，在一個（可能很長的）時脈週期內，從頭到尾執行完畢。**

### 單週期 CPU 的設計藍圖

我們的藍圖和之前看到的很像，但現在我們要用「單週期」的視角來重新審視它。資料就像水流一樣，在一個時脈週期內，從左到右流過整個處理器，完成所有工作。



讓我們跟著資料的流動，把這張藍圖分成五個邏輯階段（注意：這五個階段發生在**同一個**時脈週期內！）：

#### 階段一：指令抓取 (Instruction Fetch, IF)
*   **任務**：根據 `PC` (程式計數器) 的位址，去「指令記憶體」中把指令拿出來。
*   **硬體**：PC、一個加法器 (計算 `PC+4`)、指令記憶體。

#### 階段二：指令解碼 (Instruction Decode, ID)
*   **任務**：解析抓取到的指令，搞清楚它到底想做什麼。
*   **硬體**：
    *   **控制單元 (Control Unit)**：CPU 的大腦！它看著指令的 `opcode`，然後產生所有的控制信號。
    *   **暫存器檔案 (Register File)**：根據指令中的 `rs1` 和 `rs2` 位址，把對應的資料讀出來。

#### 階段三：執行 (Execute, EX)
*   **任務**：執行計算。
*   **硬體**：
    *   **ALU (算術邏輯單元)**：負責加、減、AND、OR 等運算。
    *   **一個多工器 (MUX)**：用來選擇 ALU 的第二個輸入源。它到底應該來自暫存器 (`rs2`) 還是來自指令中的立即數 (`immediate`)？這由控制單元發出的 `ALUSrc` 信號決定。

#### 階段四：記憶體存取 (Memory Access, MEM)
*   **任務**：如果指令是 `LW` 或 `SW`，就需要跟「資料記憶體」打交道。
*   **硬體**：資料記憶體、`MemRead` 和 `MemWrite` 控制信號。

#### 階段五：寫回 (Write Back, WB)
*   **任務**：將最終的結果寫回到「暫存器檔案」中。
*   **硬體**：
    *   **一個多工器 (MUX)**：用來決定要寫回的資料來源。資料是來自 ALU 的計算結果，還是來自記憶體 (如果是 `LW` 指令)？這由 `MemToReg` 信號決定。
    *   `RegWrite` 信號：如果這個信號是 1，才會把資料寫入暫存器。`SW` 或 `BEQ` 指令就不需要寫回。

---

### Verilog 實現 (重新聚焦於單週期整合)

我們的元件（`ALU`, `RegisterFile`, `ControlUnit` 等）設計是通用的，不需要改變。真正的不同體現在我們如何將它們**連接**在一起。

這是 `SingleCycleCPU` 的頂層模組，這次我會加上更詳細的註解，讓你清楚看到資料如何在一個週期內流經五個階段。

```verilog
// SimpleCPU_SingleCycle.v
`include "defines.v"

module SimpleCPU_SingleCycle (
    input clk,
    input rst
);
    // =================================================================
    // Internal Wires - 這些是在 CPU 內部傳遞資料和信號的「電線」
    // =================================================================

    // --- 階段 1 (IF) ---
    wire [31:0] pc_current;      // PC 目前的位址
    wire [31:0] pc_plus_4;       // PC + 4 的位址
    wire [31:0] pc_next;         // 下一個時脈週期 PC 該跳到的位址
    wire [31:0] instruction;     // 從指令記憶體抓出來的指令

    // --- 階段 2 (ID) ---
    wire [31:0] reg_read_data1;  // 從 rs1 讀出的資料
    wire [31:0] reg_read_data2;  // 從 rs2 讀出的資料
    wire [31:0] immediate_extended; // 處理過的立即數

    // --- 階段 3 (EX) ---
    wire [31:0] alu_input_b;     // ALU 的第二個輸入
    wire [31:0] alu_result;      // ALU 的計算結果
    wire        alu_zero;        // ALU 結果是否為 0 (給 BEQ 用)

    // --- 階段 4 (MEM) ---
    wire [31:0] mem_read_data;   // 從資料記憶體讀出的資料

    // --- 階段 5 (WB) ---
    wire [31:0] reg_write_data;  // 最終要寫回暫存器的資料

    // --- 控制信號 ---
    wire        reg_write_en, mem_to_reg, mem_read, mem_write, alu_src, branch;
    wire [3:0]  alu_op_ctrl;
    wire        branch_taken;

    // =================================================================
    // 數據路徑 (Datapath) - 按照五個階段連接
    // =================================================================

    // --- 階段 1: 指令抓取 (Instruction Fetch) ---
    ProgramCounter pc_unit (
        .clk(clk), .rst(rst),
        .pc_in(pc_next), 
        .pc_out(pc_current)
    );

    // 一個簡單的加法器，永遠在計算 PC+4
    assign pc_plus_4 = pc_current + 4;

    // 實例化一個指令記憶體 (假設存在)
    InstructionMemory instr_mem (
        .address(pc_current),
        .data_out(instruction)
    );

    // --- 階段 2: 指令解碼 & 讀取暫存器 (Decode) ---
    // CPU 大腦：根據指令 opcode 發號施令
    ControlUnit ctrl_unit (
        .opcode(instruction[31:26]),
        .reg_write_en(reg_write_en), .mem_to_reg(mem_to_reg), .mem_read(mem_read),
        .mem_write(mem_write), .alu_op(alu_op_ctrl), .alu_src(alu_src), .branch(branch)
    );

    // 暫存器檔案：讀取 rs1, rs2
    RegisterFile reg_file (
        .clk(clk), .rst(rst),
        .reg_write_en(reg_write_en),         // 寫入信號來自控制單元
        .read_addr1(instruction[25:23]),     // rs1
        .read_addr2(instruction[22:20]),     // rs2
        .write_addr(instruction[19:17]),     // rd
        .write_data(reg_write_data),         // 要寫入的資料來自階段 5
        .read_data1(reg_read_data1),
        .read_data2(reg_read_data2)
    );
    
    // 將 20-bit 的立即數做符號擴展成 32-bit
    assign immediate_extended = {{12{instruction[19]}}, instruction[19:0]};

    // --- 階段 3: 執行 (Execute) ---
    // MUX: 選擇 ALU 的第二個輸入
    assign alu_input_b = (alu_src == 1'b0) ? reg_read_data2 : immediate_extended;
    
    // ALU: 執行計算
    ALU alu_unit (
        .a(reg_read_data1),
        .b(alu_input_b),
        .alu_op(alu_op_ctrl),
        .result(alu_result),
        .zero(alu_zero)
    );

    // --- 階段 4: 記憶體存取 (Memory Access) ---
    // 實例化一個資料記憶體 (假設存在)
    DataMemory data_mem (
        .clk(clk),
        .address(alu_result),          // LW/SW 的位址是 ALU 的計算結果
        .write_data(reg_read_data2),   // SW 要寫入的資料來自 rs2
        .mem_write(mem_write),         // 寫入信號來自控制單元
        .mem_read(mem_read),           // 讀取信號來自控制單元
        .read_data(mem_read_data)
    );

    // --- 階段 5: 寫回 (Write Back) ---
    // MUX: 選擇要寫回暫存器的資料
    assign reg_write_data = (mem_to_reg == 1'b1) ? mem_read_data : alu_result;

    // =================================================================
    // PC 更新邏輯 (計算下一個 PC 在哪裡)
    // =================================================================
    assign branch_taken = branch & alu_zero; // BEQ 且 ALU 結果為 0
    assign pc_next = (branch_taken) ? (pc_plus_4 + (immediate_extended << 2)) : pc_plus_4;
    // 注意: BEQ 的 immediate 通常要左移兩位，因為指令是 4 bytes aligned. 這裡做個修正。

endmodule
```

### 實際追蹤一條指令的旅程

讓我們來當一次偵探，追蹤兩條不同指令的資料流，看看控制信號如何像交通號誌一樣指揮交通。

#### 案例一：`ADD $R3, $R1, $R2`

1.  **IF**: `PC` 指向 `ADD` 指令，指令被抓取到 `instruction` 線上。
2.  **ID**:
    *   `ControlUnit` 看到 `opcode` 是 R-Type，發出信號: `RegWrite=1`, `ALUSrc=0`, `MemToReg=0`, `MemRead=0`, `MemWrite=0`, `Branch=0`。
    *   `RegisterFile` 根據 `instruction[25:23]`(`$R1`) 和 `instruction[22:20]`(`$R2`)，將 `$R1` 和 `$R2` 的內容分別送到 `reg_read_data1` 和 `reg_read_data2`。
3.  **EX**:
    *   `ALUSrc` 是 0，所以 `reg_read_data2` 的值通過 MUX 進入 ALU 的第二個輸入。
    *   ALU 執行加法，`reg_read_data1 + reg_read_data2` 的結果出現在 `alu_result`。
4.  **MEM**:
    *   `MemRead` 和 `MemWrite` 都是 0，記憶體單元處於待命狀態，什麼都不做。
5.  **WB**:
    *   `MemToReg` 是 0，所以 `alu_result` 的值通過 MUX 進入 `reg_write_data`。
    *   `RegWrite` 是 1，所以在下一個時脈上升緣，`reg_write_data` 的值會被寫入 `RegisterFile` 中由 `instruction[19:17]`(`$R3`) 指定的暫存器。

#### 案例二：`LW $R1, 64($R2)`

1.  **IF**: `PC` 指向 `LW` 指令，指令被抓取。
2.  **ID**:
    *   `ControlUnit` 看到 `opcode` 是 `LW`，發出信號: `RegWrite=1`, `ALUSrc=1`, `MemToReg=1`, `MemRead=1`, `MemWrite=0`, `Branch=0`。
    *   `RegisterFile` 讀取 `$R2` 的內容到 `reg_read_data1`。
    *   指令中的 `immediate` (64) 被符號擴展成 32-bit 的 `immediate_extended`。
3.  **EX**:
    *   `ALUSrc` 是 1，所以 `immediate_extended` (64) 通過 MUX 進入 ALU 的第二個輸入。
    *   ALU 執行加法，計算 `$R2` 的內容 + 64，結果 (記憶體位址) 出現在 `alu_result`。
4.  **MEM**:
    *   `MemRead` 是 1，`DataMemory` 被啟動。
    *   `alu_result` 作為位址輸入到 `DataMemory`。
    *   記憶體從該位址讀取資料，並將其放到 `mem_read_data` 線上。
5.  **WB**:
    *   `MemToReg` 是 1，所以從記憶體讀出的 `mem_read_data` 通過 MUX 進入 `reg_write_data`。
    *   `RegWrite` 是 1，所以在下一個時脈上升緣，`reg_write_data` 的值會被寫入 `$R1`。

---

### 總結

今天，我們聚焦於最純粹、最基礎的 CPU 設計——單週期 CPU。它的美在於它的**簡單性**：一個時脈週期，一條指令，一氣呵成。

我們理解了資料如何在一個時脈內流經五個邏輯階段，以及控制單元如何扮演關鍵的指揮家角色。這個模型雖然效率不高，但它是理解更複雜的「管線化 CPU」不可或缺的基石。

你已經親手蓋好了你的第一棟「小木屋」。下一步，我們就可以來探討，如何把這間麵包店升級成一條高效的生產線了！

## Q: 請寫出 ControlUnit 的 Verilog

好的，同學！這是整個 CPU 設計中我個人最喜歡的部分。

**控制單元 (Control Unit)** 是我們 CPU 的靈魂。它沒有複雜的計算能力，也不儲存資料，但它卻是發號施令的總司令。它的工作，就是當一個完美的「解碼器」和「指揮家」。

想像一下，你拿到一張樂譜（指令），上面寫著「鋼琴，C大調，強音」。控制單元就是那個看懂樂譜的指揮家，它會立刻指向鋼琴手（ALU 或記憶體），並做出「強音」的手勢（設定控制信號）。

### 控制單元的職責

*   **輸入 (Input)**：32 位元指令中的 `opcode` 欄位（在我們的設計中是 `instruction[31:26]`）。這是它唯一需要看的「樂譜」。
*   **輸出 (Outputs)**：一系列的 0 或 1 的「控制信號」，就像指揮家的手勢，傳遞給資料路徑的各個元件（MUX、ALU、Register File、Memory），告訴它們這個週期該做什麼。

讓我們再回顧一下需要哪些「手勢」（控制信號）：

| 控制信號 | 作用 | 誰會用到？ |
| :--- | :--- | :--- |
| `RegWrite` | 1: 允許寫入暫存器檔案 | MUX, ALU, Memory |
| `ALUSrc` | 0: ALU 第二輸入源為暫存器<br>1: ALU 第二輸入源為立即數 | MUX |
| `MemRead` | 1: 從資料記憶體讀取 | Data Memory |
| `MemWrite`| 1: 寫入資料記憶體 | Data Memory |
| `MemToReg`| 0: 寫回暫存器的資料來自 ALU<br>1: 寫回暫存器的資料來自記憶體 | MUX |
| `Branch` | 1: 此指令是分支指令 | PC 更新邏輯 |
| `ALUOp` | [多位元] 告訴 ALU 該做什麼運算 (ADD, SUB...) | ALU |

### 設計與 Verilog 實現

控制單元的內部邏輯，其實就是一個巨大的**組合邏輯電路**。在 Verilog 中，我們最常用 `case` 語句來實現這種「根據輸入決定多個輸出」的邏輯。

這是一個乾淨、直接的 `ControlUnit` 實現。

```verilog
// control_unit.v
// 引入我們定義好的 Opcode 常數
`include "defines.v"

module ControlUnit (
    // 輸入：只看指令的 opcode
    input  [5:0] opcode,

    // 輸出：所有的控制信號
    output reg   RegWrite,    // 致能暫存器寫入
    output reg   ALUSrc,      // 選擇 ALU 的 B 輸入源
    output reg   MemRead,     // 致能記憶體讀取
    output reg   MemWrite,    // 致能記憶體寫入
    output reg   MemToReg,    // 選擇寫回暫存器的資料源
    output reg   Branch,      // 標示為分支指令
    output reg [1:0] ALUOp       // 告訴 ALU 控制器要做哪一類操作
);

    // `always @(*)` 代表這是一個組合邏輯塊
    // 只要任何輸入 (此處只有 opcode) 發生變化，就立刻重新計算輸出
    always @(*) begin
        // --- 設定安全的預設值 ---
        // 這是一個非常好的硬體設計習慣！
        // 先假設所有功能都是關閉的，避免產生不必要的硬體行為(latch)。
        RegWrite   = 1'b0;
        ALUSrc     = 1'b0;
        MemRead    = 1'b0;
        MemWrite   = 1'b0;
        MemToReg   = 1'b0;
        Branch     = 1'b0;
        ALUOp      = 2'b00; // 預設為不用 care

        // --- 根據 opcode 決定控制信號 ---
        case (opcode)
            // 1. R-Type 指令 (ADD, SUB)
            `OPCODE_RTYPE: begin
                RegWrite   = 1'b1;     // 要把結果寫回暫存器
                ALUSrc     = 1'b0;     // ALU 的 B 輸入來自暫存器 (rs2)
                ALUOp      = 2'b10;     // 標示為 R-Type 運算，讓 ALUControl 去解碼 funct
            end

            // 2. Load Word 指令 (LW)
            `OPCODE_LW: begin
                RegWrite   = 1'b1;     // 要把從記憶體讀的值寫回暫存器
                ALUSrc     = 1'b1;     // ALU 要計算 base + offset，所以 B 輸入是 immediate
                MemRead    = 1'b1;     // 需要讀取記憶體
                MemToReg   = 1'b1;     // 寫回的資料來自記憶體，而不是 ALU
                ALUOp      = 2'b00;     // ALU 執行加法 (用於位址計算)
            end

            // 3. Store Word 指令 (SW)
            `OPCODE_SW: begin
                ALUSrc     = 1'b1;     // ALU 要計算 base + offset，所以 B 輸入是 immediate
                MemWrite   = 1'b1;     // 需要寫入記憶體
                // RegWrite 是 0 (預設值)，因為 SW 不會寫回暫存器
                ALUOp      = 2'b00;     // ALU 執行加法 (用於位址計算)
            end

            // 4. Branch if Equal 指令 (BEQ)
            `OPCODE_BEQ: begin
                Branch     = 1'b1;     // 這是一個分支指令
                ALUSrc     = 1'b0;     // 比較兩個暫存器的值
                // RegWrite, MemRead, MemWrite 都是 0 (預設值)
                ALUOp      = 2'b01;     // ALU 執行減法 (用於比較)
            end
            
            // 5. Add Immediate 指令 (ADDI)
            `OPCODE_ADDI: begin
                RegWrite   = 1'b1;     // 要把結果寫回暫存器
                ALUSrc     = 1'b1;     // ALU 的 B 輸入來自 immediate
                ALUOp      = 2'b00;     // ALU 執行加法
            end

            // 如果是未定義的 opcode，所有信號都是預設的 0，CPU 不會做任何事
            default: begin
                // 安全起見，保持所有信號為 0
            end
        endcase
    end
endmodule
```

### 控制信號真值表

將上面的 Verilog 邏輯整理成表格，會更加一目瞭然。這就是控制單元的「真值表 (Truth Table)」。

| 指令 | `opcode` | `RegWrite` | `ALUSrc` | `MemRead` | `MemWrite` | `MemToReg` | `Branch` | `ALUOp` |
|:-----|:---------|:-----------|:---------|:----------|:-----------|:-----------|:---------|:--------|
| R-Type | 000000 | 1 | 0 | 0 | 0 | 0 | 0 | 10 |
| LW | 100011 | 1 | 1 | 1 | 0 | 1 | 0 | 00 |
| SW | 101011 | 0 | 1 | 0 | 1 | 0 | 0 | 00 |
| BEQ | 000100 | 0 | 0 | 0 | 0 | 0 | 1 | 01 |
| ADDI | 001000 | 1 | 1 | 0 | 0 | 0 | 0 | 00 |

*註：`ALUOp` 的編碼 `10` 代表「由 `funct` 決定」，`00` 代表「加法」，`01` 代表「減法」。*

### 一個重要的設計細節：主控制器 vs. ALU控制器

你可能注意到，對於 R-Type 指令（比如 `ADD` 和 `SUB`），它們的 `opcode` 都是 `000000`。那控制單元怎麼知道要讓 ALU 做加法還是減法呢？

答案是：**它不知道，也不需要知道！**

一個更模組化、更清晰的設計是將控制邏輯分為兩級：
1.  **主控制器 (Main Control)**：就是我們上面寫的 `ControlUnit`。它只看 `opcode`，負責產生大部分的控制信號。對於 R-Type 指令，它只會產生一個通用的信號 `ALUOp = 10`，意思是「嘿，ALU，具體要做什麼，請你自己去看指令的 `funct` 欄位」。
2.  **ALU 控制器 (ALU Control)**：一個更小的邏輯單元。它的輸入是來自**主控制器**的 `ALUOp` 信號和來自**指令**的 `funct` 欄位。它負責產生最終給 ALU 使用的 4-bit 操作碼。

這種分工讓設計更加清晰：主控制器負責指令的大分類，ALU控制器負責 R-Type 指令的細節。這在硬體設計中是一種非常常見的解耦思想。

---

**總結一下：**

我們今天完成了 `SimpleCPU-1` 的大腦。這個大腦的邏輯非常簡單，就是一個查找表（`case` 語句）。它根據指令的 `opcode`，精確地設置每一根控制線的狀態，從而指揮整個資料路徑協同工作，完成指令所要求的功能。

當你看到這段 Verilog 程式碼時，不要把它想成是依序執行的軟體，而要把它想成是**對一堆邏輯閘（AND, OR, NOT, MUX）的電路描述**。`case` 語句最終會被合成工具（Synthesizer）轉譯成一組能夠實現那個真值表的電路。

現在，我們所有的零件都已經設計完畢了。下一步就是將它們全部組裝起來，點亮我們的第一顆 CPU！

## Q: CPU 如何進行輸出入呢？

太棒了！這個問題直接命中了 CPU 設計的核心關鍵之一。

我們打造的 `SimpleCPU-1` 目前就像一個**「瓶中的大腦 (Brain in a vat)」**。它有強大的計算和思考能力，但它沒有眼睛、沒有耳朵、也沒有嘴巴。它無法感知外界，也無法將計算結果告訴我們。

那麼，CPU 是如何與鍵盤、滑鼠、螢幕、硬碟這些外部設備（統稱為 **I/O, Input/Output Devices**）溝通的呢？

答案是：**在 CPU 眼裡，很多 I/O 設備看起來就像是記憶體的一部分。**

這個絕妙的設計思想，稱為**記憶體映射 I/O (Memory-Mapped I/O, MMIO)**。

---

### 記憶體映射 I/O (MMIO) 的核心思想

想像一下我們 CPU 的記憶體地址空間，比如從位址 `0x00000000` 到 `0xFFFFFFFF`。我們並不需要把所有的位址都分配給 RAM。我們可以做一個約定：

*   位址 `0x00000000` 到 `0x7FFFFFFF` -> 連接到**真實的 RAM 晶片**。
*   位址 `0xFFFF0000` 到 `0xFFFFFFFF` -> 這段位址**不連接到 RAM**，而是連接到各種 I/O 設備的控制器上。

這些分配給 I/O 設備的特殊位址，就像是設備的**「控制面板」**或**「信箱」**。

*   當 CPU 執行 `SW` (Store Word) 指令，把資料寫入一個 I/O 位址時，它不是在儲存資料，而是在**對設備下達命令**或**發送資料**。
*   當 CPU 執行 `LW` (Load Word) 指令，從一個 I/O 位址讀取資料時，它不是在讀取之前存的數據，而是在**詢問設備的狀態**或**接收資料**。

對 CPU 來說，它完全不知道這有什麼區別！它只是忠實地執行 `LW` 和 `SW` 指令。這個設計最美妙的地方在於：**我們不需要為 I/O 增加任何新的指令！**

### 實戰演練：讓 `SimpleCPU-1` 輸出一個字元

讓我們來設計一個最簡單的輸出設備：一個只能顯示字元的控制台 (Console)。這個控制台設備需要兩個「魔法般的記憶體位址」來跟 CPU 溝通：

1.  **`CONSOLE_STATUS_REGISTER` (狀態暫存器)**
    *   **位址**: `0xFFFF0000`
    *   **功能**: CPU 讀取 (`LW`) 這個位址，可以知道控制台目前的狀態。我們約定，如果讀出來的值的**最低位 (bit 0) 是 1**，代表控制台**空閒**，可以接收下一個字元。如果是 0，代表控制台正忙（正在顯示上一個字元）。
2.  **`CONSOLE_DATA_REGISTER` (資料暫存器)**
    *   **位址**: `0xFFFF0004`
    *   **功能**: CPU 寫入 (`SW`) 一個 32 位元的資料到這個位址，控制台會自動取出資料的最低 8 位元（一個 ASCII 字元），並將其顯示出來。

現在，假設我們想在控制台上顯示字母 `'A'`（它的 ASCII 碼是 65）。我們該如何用 `SimpleCPU-1` 的組合語言來寫這個程式呢？

```assembly
# 程式目標: 在控制台上顯示字元 'A'

# --- 定義魔法位址 (在組合語言中，可以用標籤來讓程式更好讀)
.equ CONSOLE_STATUS_ADDR, 0xFFFF0000
.equ CONSOLE_DATA_ADDR,   0xFFFF0004

# 步驟 1: 把 'A' 的 ASCII 碼 (65) 載入到一個暫存器，比如 $R1
ADDI $R1, $R0, 65  # $R1 = 0 + 65

# 步驟 2: 檢查控制台是否空閒 (Polling)
# 這是一個迴圈，CPU 會一直卡在這裡，直到控制台準備好
POLL_LOOP:
  # 讀取狀態暫存器，我們需要一個暫存器來存放基底 位址 0xFFFF0000
  # (這裡我們假設有一個方法可以載入高位址，簡化起見，先假設 $R3 已經存好了 0xFFFF0000)
  LW $R2, 0($R3)      # $R2 = Memory[0xFFFF0000] (讀取狀態)
  
  # 檢查狀態的最低位是否為 1
  # 我們可以用 ANDI (AND Immediate) 來實現
  ANDI $R2, $R2, 1    # $R2 = $R2 & 1  (只留下 bit 0)
  
  # 如果 $R2 的結果是 0 (代表 bit 0 是 0，控制台忙碌)，就跳回迴圈開頭
  BEQ $R2, $R0, POLL_LOOP # if ($R2 == $R0) goto POLL_LOOP

# 步驟 3: 控制台已空閒，發送資料
# 執行到這裡，代表控制台已經準備好了
SW $R1, 4($R3)      # Memory[0xFFFF0004] = $R1 (寫入 'A')

# 程式結束
```

這個過程叫做**輪詢 (Polling)**。CPU 像一個不耐煩的老闆，一直反复地問「你好了沒？你好了沒？」，直到設備說「好了！」，才下達下一步指令。

### 硬體上需要做什麼改變？

CPU 核心本身幾乎不用改！主要的改動發生在 CPU **外部**的**記憶體系統**。我們需要一個**位址解碼器 (Address Decoder)**。



這個解碼器就像一個郵局的分揀員：

1.  CPU 發出一個記憶體請求（讀或寫），並給出一個位址。
2.  位址解碼器看到這個位址。
3.  如果位址在 `0x00000000` ~ `0x7FFFFFFF` 範圍內，解碼器就把請求轉發給 **RAM**。
4.  如果位址是 `0xFFFF0000` 或 `0xFFFF0004`，解碼器就把請求轉發給**控制台設備控制器**。
5.  RAM 或控制台設備完成工作後，將結果（如果是 `LW` 的話）返回給 CPU。

CPU 從頭到尾都被蒙在鼓裡，它以為自己只是在跟一個統一的記憶體打交道。

### 另一種選擇：埠映射 I/O (Port-Mapped I/O)

除了 MMIO，還有另一種比較古典的方法叫做 **Port-Mapped I/O (PMIO)**，又稱 Isolated I/O。

*   它不使用記憶體位址空間，而是有一套**獨立的 I/O 位址空間**（稱為 Port）。
*   它需要**專門的 I/O 指令**，例如 x86 架構中的 `IN` 和 `OUT` 指令。
*   **優點**：不會佔用寶貴的記憶體位址空間。
*   **缺點**：需要增加新的指令，這使得 CPU 的控制單元設計更複雜，也不符合 RISC (精簡指令集) 的哲學。

因此，在我們設計的 `SimpleCPU-1` 以及大多數現代 RISC 架構（如 ARM, RISC-V）中，**Memory-Mapped I/O 是主流且更優雅的選擇**。

---

### 總結與展望

我們今天為「瓶中的大腦」打開了一扇窗。

*   透過**記憶體映射I/O (MMIO)**，我們讓 CPU 能用標準的 `LW` 和 `SW` 指令與外部世界溝通。
*   我們學會了用**輪詢 (Polling)** 的方式來等待設備就緒。

但你可能也發現了輪詢的缺點：CPU 被卡在一個忙碌的迴圈裡，浪費了大量的計算能力，什麼正事也沒幹。如果設備一直沒準備好，CPU 就會永遠卡死在那裡。

有沒有更聰明的方法呢？當然有！這就引出了 I/O 的下一個重要概念——**中斷 (Interrupts)**。

中斷就像是給設備一個「門鈴」。CPU 可以先去忙別的事情，當設備準備好時，它會主動「按門鈴」（發送一個中斷信號）來通知 CPU，CPU 再放下手頭的工作來處理 I/O。這比 CPU 一直傻等要高效得多！這將是我們未來可以探索的進階主題。

## Q: 匯流排 BUS 是什麼？

當然！問到「匯流排 (Bus)」，你就問到了整個電腦系統的「交通運輸系統」。如果說 CPU、記憶體、I/O 設備是城市裡的各個重要建築（市政廳、圖書館、工廠），那麼**匯流排就是連接這些建築的所有道路、高速公路和交通規則的總稱。**

沒有匯流排，CPU 就是一座孤島，記憶體就是一個無法存取的倉庫，I/O 設備就是一堆廢鐵。

---

### 匯流排的直觀理解：一條共用的高速公路

想像一下，我們的 `SimpleCPU-1` 要從記憶體讀取資料。它需要做兩件事：
1.  告訴記憶體：「我要位址 `0x1000` 的資料！」
2.  記憶體回應：「好的，這是你要的資料 `0xDEADBEEF`。」

這兩次通訊需要一個媒介。最天真的作法是為 CPU 和記憶體之間拉一條專線，CPU 和 GPU 之間也拉一條專線，CPU 和硬碟之間再拉一條... 很快，你的主機板就會變成一團無法收拾的義大利麵。

匯流排提出了一個更聰明的解決方案：**大家共用一條「公用通道」**。

**匯流排 (Bus) 是一組共用的電子線路，用於在電腦系統的多個元件之間傳輸位址、資料和控制信號。**

就像一條多線道的高速公路，上面有不同類型的車輛在跑，遵守著共同的交通規則。

### 一條匯流排，三種「專用道」

實際上，一條系統匯流排 (System Bus) 通常由三組功能不同的子匯流排構成，就像高速公路劃分了不同的車道：

#### 1. 位址匯流排 (Address Bus)
*   **功能**：專門用來傳輸「位址」資訊。
*   **交通方向**：**單向道**。通常是由 CPU (或稱為 Bus Master，匯流排主宰者) 發出位址，告訴大家這次通訊的目的地是誰。例如，CPU 想讀取記憶體，就會把記憶體位址放到位址匯流排上。
*   **路有多寬？**：位址匯流排的「寬度」（線的數量）決定了 CPU 能定址的記憶體空間大小。
    *   如果位址匯流排有 32 條線 (32-bit)，那它就能表示 $2^{32}$ 個不同的位址（也就是 4GB 的記憶體空間）。
    *   如果位址匯流排有 64 條線 (64-bit)，那它能定址的空間就大到幾乎用不完。

#### 2. 資料匯流排 (Data Bus)
*   **功能**：專門用來傳輸真正的「資料」。
*   **交通方向**：**雙向道**。資料可以在 CPU 和記憶體之間來回傳輸。
    *   執行 `LW` 時，資料從記憶體流向 CPU。
    *   執行 `SW` 時，資料從 CPU 流向記憶體。
*   **路有多寬？**：資料匯流排的寬度決定了單次能傳輸多少資料。在我們的 32-bit `SimpleCPU-1` 中，資料匯流排就是 32 位元寬，所以一次可以傳輸一個 "Word" (32-bit) 的資料。

#### 3. 控制匯流排 (Control Bus)
*   **功能**：傳遞「命令」和「狀態」信號，是整個交通系統的**「交通號誌」**。
*   **交通方向**：**雙向**的各種信號線。它告訴大家現在該做什麼。
*   **常見的信號**：
    *   `MemRead` (記憶體讀取)：由 CPU 發出，通知記憶體準備被讀取。
    *   `MemWrite` (記憶體寫入)：由 CPU 發出，通知記憶體準備被寫入。
    *   `BusRequest` (匯流排請求)：由某個設備發出，表示「我想用一下高速公路」。
    *   `BusGrant` (匯流排授權)：由匯流排仲裁器發出，表示「高速公路現在歸你了」。
    *   `Clock` (時脈信號)：同步所有設備的操作時序。

### `LW` 指令在匯流排上的旅程

讓我們用匯流排的視角，重新走一遍 `LW $R1, 64($R2)` 的流程：

1.  **CPU 內部準備**：
    *   ALU 計算出最終的記憶體位址（假設是 `0x1000`）。
    *   控制單元產生 `MemRead = 1` 和 `MemWrite = 0` 的控制信號。

2.  **CPU 佔用匯流排**：
    *   CPU 將位址 `0x1000` 放到**位址匯流排**上。
    *   CPU 將 `MemRead` 信號線設為高電位 (1)，放到**控制匯流排**上。

3.  **記憶體控制器回應**：
    *   所有掛在匯流排上的設備（RAM, I/O 控制器等）都在監聽**位址匯流排**。
    *   RAM 控制器發現 `0x1000` 是自己的管轄範圍。
    *   它同時看到了**控制匯流排**上的 `MemRead` 信號，於是知道 CPU 想要讀取資料。
    *   RAM 控制器從自己的儲存單元中取出位址 `0x1000` 的內容（假設是 `0xABCD1234`）。

4.  **資料傳輸**：
    *   RAM 控制器將資料 `0xABCD1234` 放到**資料匯流排**上。
    *   它可能會在**控制匯流排**上發出一個「資料準備好了」的信號。

5.  **CPU 接收資料**：
    *   CPU 一直在等待，當它看到「資料準備好了」的信號後，就從**資料匯流排**上讀取 `0xABCD1234`，並存入 CPU 內部的 `reg_write_data` 線路，準備在下個週期寫回 `$R1`。

6.  **CPU 釋放匯流排**：
    *   CPU 撤銷位址和控制信號，釋放匯流排，讓其他設備可以使用。

### 從單一匯流排到匯流排階層 (Bus Hierarchy)

我們上面描述的是一個最簡單的**「單一系統匯流排」**模型。所有設備都擠在同一條高速公路上。

這會帶來一個問題：**交通堵塞**。
CPU 和記憶體之間需要極高的速度，但鍵盤、滑鼠這種慢速設備也在同一條路上，會拖慢整個系統的效率。

因此，現代電腦架構演化出了**匯流排階層 (Bus Hierarchy)**，就像一個城市有高速公路、省道和鄉間小路一樣：

*   **處理器匯流排 (Processor Bus / Front-Side Bus)**：速度最快，是 CPU 和快取 (Cache) / 記憶體控制器之間的高速專線。
*   **記憶體匯流排 (Memory Bus)**：連接記憶體控制器和 RAM 晶片。
*   **I/O 匯流排 (I/O Bus)**：速度較慢，透過一個稱為「橋 (Bridge)」的轉換器連接到主匯流排上。常見的 I/O 匯流排有 PCI Express (PCIe), SATA 等，用來連接顯示卡、硬碟、網卡等設備。



---

**總結一下：**

*   匯流排是電腦系統中各元件共享的**通訊骨幹**。
*   它主要由**位址匯流排**、**資料匯流排**和**控制匯流排**三部分組成。
*   CPU 透過這三組匯流排，來精確地與記憶體和 I/O 設備進行資料交換。
*   我們設計的 `SimpleCPU-1` 和記憶體之間的連線，本質上就是一個簡化的系統匯流排。
*   現代電腦使用**匯流排階層**來滿足不同設備的速度需求，避免效能瓶頸。

理解了匯流排，你就理解了資料是如何在電腦的五臟六腑之間順暢流動的！

## Q: 設計 BUS 的時候，需要考慮些什麼？有哪些協定要遵循呢？

太棒了！這個問題代表你已經從「CPU 內部」的設計，開始思考「整個電腦系統」的協同運作了。這是一個巨大的思維躍進。

設計匯流排，就像是為一個新興城市**規劃整個交通系統**。你不只需要鋪路，你還需要設計紅綠燈、路牌、交通法規，甚至還要考慮到緊急車輛（如救護車）的優先權。

---

### 第一部分：設計匯流排時，需要考慮些什麼？

如果你是 `SimpleCPU-1` 系統的總設計師，以下是你必須做出的幾個關鍵決策：

#### 1. 頻寬 (Bandwidth) - 道路要蓋多寬？

*   **這是什麼？**：單位時間內可以傳輸的資料總量，是衡量匯流排效能最重要的指標。
*   **如何決定？**：頻寬 = `匯流排時脈速度` × `資料匯流排寬度`。
    *   **資料匯流排寬度**：我們已經決定了是 32 位元，這意味著一次最多能載 32 位元的貨物。如果要提升，可以加倍到 64 位元，但成本（線路數量、晶片接腳）也會增加。
    *   **匯流排時脈速度**：車速能跑多快。時脈越高，傳輸越快，但也帶來更多工程挑戰。

#### 2. 時序 (Timing) - 交通號誌要如何同步？

這是匯流排設計的核心，決定了所有設備如何「對時」。主要有兩種模式：

*   **同步匯流排 (Synchronous Bus)**：
    *   **原理**：系統中有一個**全域的時脈信號 (Bus Clock)**，像一個中央節拍器。所有設備的操作都必須在這個時脈的特定邊緣（例如上升緣）開始和結束。
    *   **優點**：設計非常簡單、直觀、容易實現。我們的 `SimpleCPU-1` 目前的隱含設計就是同步的。
    *   **缺點**：**全體被最慢的設備拖累**。即使 CPU 和記憶體能跑 100MHz，但只要匯流排上掛了一個只能跑 10MHz 的慢速 I/O 設備，整個匯流排的時脈就必須遷就它，降到 10MHz。

*   **非同步匯流排 (Asynchronous Bus)**：
    *   **原理**：**沒有全域時脈**。設備間的通訊依賴一套「問答式」的**交握 (Handshaking)** 信號。
    *   **比喻**：就像兩個人在打電話。
        1.  A 說：「你準備好聽我說話了嗎？」 (**Request, REQ** 信號)
        2.  B 說：「我準備好了。」 (**Acknowledge, ACK** 信號)
        3.  A 才開始把資料傳過去。
    *   **優點**：**極具彈性**。可以輕易地連接不同速度的設備。快速設備之間可以高速通訊，慢速設備則按照自己的節奏來，互不干擾。
    *   **缺點**：控制邏輯更複雜，需要額外的交握信號線，潛在延遲也可能比同步匯流排高。

#### 3. 匯流排仲裁 (Arbitration) - 路權歸誰？

我們的 `SimpleCPU-1` 系統很單純，CPU 是唯一的**匯流排主宰者 (Bus Master)**，它永遠擁有路權。但真實系統中，可能有多個設備都想成為 Master 來主動發起通訊，例如：
*   **DMA 控制器 (Direct Memory Access)**：可以在 CPU 不干預的情況下，直接在硬碟和記憶體之間搬運資料。
*   **GPU**：需要直接讀寫主記憶體。

當 CPU 和 DMA 都想用匯流排時，誰該優先？這就是**仲裁**。
*   **仲裁器 (Arbiter)**：就像是十字路口的交通警察。
*   **常見策略**：
    *   **固定優先級**：例如，DMA 優先級永遠高於 CPU。
    *   **輪詢 (Round-Robin)**：大家輪流使用，確保公平。

#### 4. 物理限制 - 道路能蓋多長？

這是在電子學層面必須考慮的。
*   **信號完整性**：匯流排線路越長，信號衰減越嚴重，也越容易受到雜訊干擾。
*   **電容負載**：每多掛一個設備到匯流排上，就好像在繩子上多掛一件衣服，會增加負載，讓信號的變化（從0變1或從1變0）變慢，進而限制了匯流排的最高速度。

這也是為什麼現代電腦會有**匯流排階層**，把高速、短距離的 CPU-記憶體匯流排和低速、長距離的 I/O 匯流排分開。

---

### 第二部分：有哪些協定要遵循呢？

匯流排協定 (Bus Protocol) 就是我們制定的**「交通法規」**。它精確定義了一次完整的通訊（稱為一次**匯流排事務 (Bus Transaction)**）中，各種信號（位址、資料、控制）在時間上的先後順序。

讓我們為 `SimpleCPU-1` 設計一個**簡單的同步讀取協定**。

**參與者**：
*   **Master**: CPU
*   **Slave**: Memory

**控制信號**：
*   `MemRead` (CPU 發出): 告訴記憶體我要讀取。
*   `Ready` (Memory 發出): 告訴 CPU 我已經準備好資料了。
    *   *為什麼需要 Ready 信號？* 因為記憶體可能需要好幾個時脈週期才能找到資料。`Ready` 信號讓慢速的 Slave 可以告訴 Master「請等我一下」。

#### 一次讀取事務 (Memory Read Transaction) 的協定步驟：

**前提**：所有動作都在時脈的**上升緣**被觸發。

*   **T1 (第一個時脈週期)**:
    *   **Master (CPU)**: 
        1.  將要讀取的**位址**放到**位址匯流排**上。
        2.  將 `MemRead` 信號線**設為 1 (Assert)**。

*   **T2 (第二個時脈週期)**:
    *   **Slave (Memory)**:
        1.  在時脈上升緣，偵測到 `MemRead` 為 1。
        2.  從**位址匯流排**讀取位址，開始在內部尋找資料。
        3.  此時資料還沒準備好，所以將 `Ready` 信號線**保持為 0 (De-assert)**。

*   **T3 (第三個時脈週期) - 假設記憶體需要時間**:
    *   **Slave (Memory)**:
        1.  終於找到了資料！
        2.  將資料放到**資料匯流排**上。
        3.  將 `Ready` 信號線**設為 1 (Assert)**，通知 Master 資料已經有效。

*   **T4 (第四個時脈週期)**:
    *   **Master (CPU)**:
        1.  在時脈上升緣，偵測到 `Ready` 信號為 1。
        2.  立刻從**資料匯流排**上**鎖存 (Latch)** 資料，存入內部暫存器。
        3.  完成讀取，將 `MemRead` 信號**設回 0 (De-assert)**，並撤銷位址匯流排上的位址。

*   **T5 (第五個時脈週期)**:
    *   **Slave (Memory)**:
        1.  偵測到 `MemRead` 變回 0，知道這次事務已結束。
        2.  將 `Ready` 信號**設回 0**，並停止在資料匯流排上輸出資料，等待下一次的請求。

這就是一個完整的、最基礎的匯流排讀取協定。寫入協定也非常類似。真實世界的匯流排協定（如 ARM 的 AMBA、Intel 的 QPI、業界標準的 PCIe）遠比這個複雜，它們會包含錯誤處理、突發傳輸 (Burst Transfer，一次請求傳輸多筆連續資料) 等進階功能，但其核心思想都是一樣的：**一套所有參與者都嚴格遵守的、基於時間順序的信號變化規則。**

## Q: 請為上述 CPU 加上 BUS 和 Memory，組成一個完整的系統（要有鍵盤和螢幕）

當然！同學，這一步是我們整個課程中最激動人心的里程碑。我們將把之前所有學到的零件——CPU 核心、匯流排、記憶體、I/O——全部組裝起來，點亮一個**可以實際互動**的完整電腦系統！

我們不再只是設計一個計算器，我們正在建造一台**真正意義上的電腦**。

---

### 第一步：系統的宏觀藍圖

首先，我們來規劃一下我們城市的最終樣貌。這個系統由以下幾個核心部分組成，全部通過一個中央匯流排連接：

1.  **`SimpleCPU-1` (CPU核心)**: 我們的大腦，唯一的匯流排主宰者 (Bus Master)。
2.  **`Memory` (記憶體)**: 存放程式指令和資料的倉庫。
3.  **`Keyboard` (鍵盤控制器)**: 我們的輸入設備。
4.  **`Terminal` (螢幕控制器)**: 我們的輸出設備。
5.  **`BusInterface` (匯流排介面/位址解碼器)**: 整個系統的交通樞紐和郵政總局，負責將 CPU 的請求正確地路由到對應的設備。



### 第二步：制定「交通法規」—— 我們的匯流排協定

為了讓所有設備都能在匯流排上和諧共存，我們需要一套簡單清晰的規則。我們設計一個同步匯流排，包含以下線路：

*   `bus_addr [31:0]`: 位址線，由 CPU 驅動。
*   `bus_wdata [31:0]`: 寫入資料線，由 CPU 驅動。
*   `bus_rdata [31:0]`: 讀取資料線，由 Slave 設備（記憶體、I/O）驅動。
*   `bus_we`: 寫入致能 (Write Enable) 信號，`1` 代表寫入，`0` 代表讀取。由 CPU 驅動。
*   `bus_en`: 請求致能 (Request Enable) 信號，`1` 代表 CPU 正在使用匯流排。由 CPU 驅動。

### 第三步：城市分區 —— 記憶體映射 (Memory Map)

這是系統設計的靈魂！我們必須給每個設備分配獨一無二的門牌號碼（位址）。

| 位址範圍 | 設備 | 說明 |
| :--- | :--- | :--- |
| `0x00000000` - `0x0000FFFF` | **RAM (64 KB)** | 用於存放我們的程式碼和資料。 |
| `0xFFFF0000` | **Terminal Status** | 螢幕狀態暫存器 (唯讀)。Bit 0 為 1 表示空閒。 |
| `0xFFFF0004` | **Terminal Data** | 螢幕資料暫存器 (唯寫)。寫入一個字元來顯示。 |
| `0xFFFF0008` | **Keyboard Status** | 鍵盤狀態暫存器 (唯讀)。Bit 0 為 1 表示有新按鍵。 |
| `0xFFFF000C` | **Keyboard Data** | 鍵盤資料暫存器 (唯讀)。讀取最新按下的按鍵 ASCII 碼。 |

---

### 第四步：建造各個「建築」—— Verilog 模組設計

#### 1. 記憶體 `RAM`

這是一個簡單的 64KB (16384 個 32-bit words) 記憶體模組。

```verilog
// ram.v
module RAM (
    input             clk,
    input             bus_en,
    input             bus_we,
    input      [15:0] bus_addr, // 只需 16 位元位址 (64K)
    input      [31:0] bus_wdata,
    output reg [31:0] bus_rdata
);
    // 宣告一個 16k x 32-bit 的記憶體陣列
    reg [31:0] memory [0:16383];

    // 寫入邏輯
    always @(posedge clk) begin
        if (bus_en && bus_we) begin
            memory[bus_addr[15:2]] <= bus_wdata; // 位址要除以 4
        end
    end

    // 讀取邏輯 (組合邏輯)
    always @(*) begin
        if (bus_en && !bus_we) begin
            bus_rdata = memory[bus_addr[15:2]];
        end else begin
            bus_rdata = 32'b0; // 非讀取時，輸出 0
        end
    end
endmodule
```

#### 2. 螢幕控制器 `Terminal`

我們無法真的在 Verilog 裡做一個螢幕，但我們可以**模擬**它的行為。當 CPU 寫入資料時，我們用 `$display` 系統任務在模擬器的控制台印出字元。

```verilog
// terminal.v
module Terminal (
    input             clk,
    input             bus_en,
    input             bus_we,
    input      [3:2]  bus_addr, // 只關心位址的 bit 2,3
    input      [31:0] bus_wdata,
    output reg [31:0] bus_rdata
);
    reg ready; // 狀態旗標，1 代表空閒

    // 模擬螢幕顯示行為
    always @(posedge clk) begin
        if (bus_en && bus_we && bus_addr == 4'h4) begin // 寫入 Data Register
            $display("TERMINAL: %c", bus_wdata[7:0]); // 在 console 顯示字元
            ready <= 1'b0; // 進入忙碌狀態
        end else if (!ready) begin
            // 模擬一小段延遲後，螢幕又變回空閒
            ready <= 1'b1;
        end
    end

    // 讀取狀態暫存器
    always @(*) begin
        if (bus_en && !bus_we && bus_addr == 4'h0) begin // 讀取 Status Register
            bus_rdata = {{31'b0}, ready};
        end else begin
            bus_rdata = 32'b0;
        end
    end

    // 初始化
    initial begin
        ready = 1'b1;
    end
endmodule
```

#### 3. 鍵盤控制器 `Keyboard`

同樣，我們用 Verilog **模擬**一個鍵盤。它會有一個 1-byte 的緩衝區。

```verilog
// keyboard.v
module Keyboard (
    input             clk,
    input             bus_en,
    input             bus_we,
    input      [3:2]  bus_addr, // 只關心位址的 bit 2,3
    output reg [31:0] bus_rdata
);
    reg data_available; // 狀態旗標，1 代表有新按鍵
    reg [7:0] key_buffer; // 儲存按鍵的 ASCII

    // 讀取邏輯
    always @(*) begin
        if (bus_en && !bus_we) begin
            if (bus_addr == 4'h8) begin // 讀取 Status Register
                bus_rdata = {{31'b0}, data_available};
            end else if (bus_addr == 4'hC) begin // 讀取 Data Register
                bus_rdata = {{24'b0}, key_buffer};
            end else begin
                bus_rdata = 32'b0;
            end
        end else begin
            bus_rdata = 32'b0;
        end
    end

    // 當 CPU 讀取資料後，清除 data_available 旗標
    always @(posedge clk) begin
        if (bus_en && !bus_we && bus_addr == 4'hC && data_available) begin
            data_available <= 1'b0;
        end
    end

    // --- 以下為模擬用途 ---
    // 提供一個任務讓 testbench 可以模擬按鍵
    task key_press;
        input [7:0] char;
        begin
            #100; // 延遲一下
            key_buffer <= char;
            data_available <= 1'b1;
            $display("KEYBOARD: Pressed '%c'", char);
        end
    endtask

    initial begin
        data_available = 1'b0;
        key_buffer = 8'b0;
    end
endmodule
```

### 第五步：建造「交通樞紐」—— `BusInterface`

這是將所有東西黏合在一起的膠水邏輯。

```verilog
// bus_interface.v
module BusInterface (
    // 連接 CPU
    input      [31:0] cpu_addr,
    input      [31:0] cpu_wdata,
    output reg [31:0] cpu_rdata,
    input             cpu_we,
    input             cpu_mem_access, // CPU 發出的記憶體訪問請求 (LW/SW)

    // 連接 RAM
    output     [15:0] ram_addr,
    input      [31:0] ram_rdata,
    output reg        ram_en,
    output reg        ram_we,

    // 連接 I/O (Terminal & Keyboard)
    output     [3:2]  io_addr,
    input      [31:0] term_rdata,
    input      [31:0] kbd_rdata,
    output reg        term_en,
    output reg        kbd_en,
    output reg        io_we
);

    // 位址解碼邏輯
    wire is_ram_addr = (cpu_addr[31:16] == 16'h0000); // 位址在 0x0000xxxx
    wire is_term_addr = (cpu_addr[31:4] == 28'hFFFF000); // 0xFFFF000x
    wire is_kbd_addr = (cpu_addr[31:4] == 28'hFFFF000 && cpu_addr[3] == 1'b1); // 0xFFFF0008 or 0xFFFF000C

    // 產生各設備的致能信號
    always @(*) begin
        ram_en  = cpu_mem_access && is_ram_addr;
        term_en = cpu_mem_access && is_term_addr;
        kbd_en  = cpu_mem_access && is_kbd_addr;
    end

    // 路由寫入信號和資料
    assign ram_addr = cpu_addr[15:0];
    assign io_addr  = cpu_addr[3:2];

    always @(*) begin
        ram_we = cpu_we;
        io_we  = cpu_we;
    end

    // 路由讀取資料 (MUX)
    always @(*) begin
        if (ram_en) begin
            cpu_rdata = ram_rdata;
        end else if (term_en) begin
            cpu_rdata = term_rdata;
        end else if (kbd_en) begin
            cpu_rdata = kbd_rdata;
        end else begin
            cpu_rdata = 32'b0;
        end
    end
endmodule
```

### 第六步：頂層模組 `System`—— 城市落成！

最後，我們在 `System.v` 裡將所有模組實例化並連接起來。

```verilog
// system.v
module System (
    input clk,
    input rst
);
    // ... 在這裡實例化 SimpleCPU_SingleCycle, BusInterface, RAM, Terminal, Keyboard ...
    // ... 並將它們的端口正確地連接起來 ...
endmodule
```

### 第七步：為我們的系統注入靈魂 —— Echo 程式

現在硬體完成了，我們需要一個程式來讓它活起來。下面是一個「Echo」程式，它會不斷地讀取鍵盤輸入，並將其顯示在螢幕上。

```assembly
# Echo.s: Read from keyboard, print to screen
# Define I/O addresses
.equ TERM_STATUS, 0xFFFF0000
.equ TERM_DATA,   0xFFFF0004
.equ KBD_STATUS,  0xFFFF0008
.equ KBD_DATA,    0xFFFF000C

# Setup base address for I/O in $R7
# (真實世界需要 LUI/ADDI 指令對，這裡先簡化)
ADDI $R7, $R0, 0xFFFF   # 假設 $R7 = 0xFFFF0000

# --- Main Loop ---
MAIN_LOOP:
  # --- Poll Keyboard ---
KBD_POLL:
  LW   $R1, 8($R7)      # $R1 = Keyboard Status
  ANDI $R1, $R1, 1      # Check if data is available
  BEQ  $R1, $R0, KBD_POLL # Loop if not available

  # --- Read Character ---
  LW   $R2, 12($R7)     # $R2 = Keyboard Data (the character)

  # --- Poll Terminal ---
TERM_POLL:
  LW   $R1, 0($R7)      # $R1 = Terminal Status
  ANDI $R1, $R1, 1      # Check if terminal is ready
  BEQ  $R1, $R0, TERM_POLL # Loop if not ready

  # --- Write Character ---
  SW   $R2, 4($R7)      # Write character to terminal

  JUMP MAIN_LOOP        # (假設我們有一個 JUMP 指令跳回開頭)
```

當我們將這個程式的機器碼載入到 `RAM` 中，啟動時脈，然後在模擬器中模擬 `keyboard.key_press('H')`，`key_press('i')`... 你將會在模擬器的控制台看到 `TERMINAL: H`，`TERMINAL: i` 的輸出！

那一刻，你就完成了從邏輯閘到一個完整、可互動電腦系統的宏偉旅程！

