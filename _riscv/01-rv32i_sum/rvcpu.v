`timescale 1ns / 1ps

// --- 頂層模組 (Top Module) ---
module riscv_mcu #(
    parameter HEX_FILE = "sum.hex"
)(
    input clk,
    input rst_n,
    output [31:0] result_a0 // 用於觀察結果 (x10)
);
    // 內部訊號
    wire [31:0] pc, pc_next, instruction;
    wire [31:0] alu_out, reg_rdata1, reg_rdata2, imm_ext;
    wire [31:0] src_b; // ALU 的第二個輸入
    wire reg_write, alu_src, branch, jump;
    wire [2:0] alu_ctrl;
    wire zero_flag, lt_flag; // Zero and Less-Than flags
    wire pc_src;

    // --- 1. Program Counter (PC) ---
    reg [31:0] pc_reg;
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) pc_reg <= 0;
        else pc_reg <= pc_next;
    end
    assign pc = pc_reg;

    // --- 2. Instruction Memory (IMEM) ---
    instruction_memory #(.FILENAME(HEX_FILE)) imem (
        .addr(pc),
        .inst(instruction)
    );

    // --- 3. Control Unit (Decoder) ---
    control_unit ctrl (
        .opcode(instruction[6:0]),
        .funct3(instruction[14:12]),
        .funct7(instruction[31:25]),
        .reg_write(reg_write),
        .alu_src(alu_src),
        .alu_ctrl(alu_ctrl),
        .branch(branch),
        .jump(jump)
    );

    // --- 4. Register File ---
    register_file rf (
        .clk(clk),
        .rst_n(rst_n),
        .we(reg_write),
        .raddr1(instruction[19:15]), // rs1
        .raddr2(instruction[24:20]), // rs2
        .waddr(instruction[11:7]),   // rd
        .wdata(alu_out),             // 簡化：假設所有運算結果都來自 ALU (無 Load 指令)
        .rdata1(reg_rdata1),
        .rdata2(reg_rdata2),
        .monitor_a0(result_a0)       // Debug port
    );

    // --- 5. Immediate Generator ---
    imm_gen ig (
        .inst(instruction),
        .imm_ext(imm_ext)
    );

    // --- 6. ALU & Muxes ---
    assign src_b = (alu_src) ? imm_ext : reg_rdata2; // Mux for ALU input B

    alu main_alu (
        .src_a(reg_rdata1),
        .src_b(src_b),
        .ctrl(alu_ctrl),
        .result(alu_out),
        .zero(zero_flag),
        .lt(lt_flag)
    );

    // --- 7. Next PC Logic ---
    // 判斷分支：如果是 Branch 指令且條件成立 (BLT)，或者如果是 JUMP (JAL)
    // 注意：上一題的 bgt 被轉為 blt，所以我們檢查 lt_flag
    // Branch logic for BLT: if (branch && lt_flag) take branch
    assign pc_src = (branch & lt_flag) | jump;
    
    // PC + 4 or PC + Imm
    assign pc_next = (pc_src) ? (pc + imm_ext) : (pc + 4);

endmodule


// --- 子模組定義 ---

// 1. 指令記憶體
module instruction_memory #(parameter FILENAME = "sum.hex")(
    input [31:0] addr,
    output [31:0] inst
);
    reg [31:0] mem [0:255]; // 256 words memory

    initial begin
        // 這裡會讀取 hex 檔案
        $readmemh(FILENAME, mem);
    end

    // Word align address (addr / 4)
    assign inst = mem[addr[31:2]]; 
endmodule

// 2. 暫存器檔案
module register_file(
    input clk, rst_n, we,
    input [4:0] raddr1, raddr2, waddr,
    input [31:0] wdata,
    output [31:0] rdata1, rdata2,
    output [31:0] monitor_a0 // 直接輸出 x10 (a0) 方便觀察
);
    reg [31:0] regs [0:31];
    integer i;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<32; i=i+1) regs[i] <= 0;
        end else if (we && waddr != 0) begin
            regs[waddr] <= wdata;
        end
    end

    assign rdata1 = (raddr1 == 0) ? 0 : regs[raddr1];
    assign rdata2 = (raddr2 == 0) ? 0 : regs[raddr2];
    assign monitor_a0 = regs[10]; 
endmodule

// 3. 立即值生成器
module imm_gen(
    input [31:0] inst,
    output reg [31:0] imm_ext
);
    always @(*) begin
        case (inst[6:0])
            7'b0010011: imm_ext = {{20{inst[31]}}, inst[31:20]};             // I-Type (addi)
            7'b1100011: imm_ext = {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0}; // B-Type (beq/blt)
            7'b1101111: imm_ext = {{11{inst[31]}}, inst[31], inst[19:12], inst[20], inst[30:21], 1'b0}; // J-Type (jal)
            default:    imm_ext = 0;
        endcase
    end
endmodule

// 4. 控制單元 (簡化版，僅支援本次作業需要的指令)
module control_unit(
    input [6:0] opcode,
    input [2:0] funct3,
    input [6:0] funct7,
    output reg reg_write, alu_src, branch, jump,
    output reg [2:0] alu_ctrl
);
    always @(*) begin
        // Defaults
        reg_write = 0; alu_src = 0; branch = 0; jump = 0; alu_ctrl = 0;

        case (opcode)
            7'b0110011: begin // R-Type (add)
                reg_write = 1;
                alu_ctrl = 3'b000; // ADD
            end
            7'b0010011: begin // I-Type (addi)
                reg_write = 1;
                alu_src = 1;       // Use Immediate
                alu_ctrl = 3'b000; // ADD
            end
            7'b1100011: begin // B-Type (blt)
                branch = 1;
                alu_ctrl = 3'b010; // SLT (Set Less Than) for comparison
            end
            7'b1101111: begin // J-Type (jal)
                jump = 1;
                // JAL 通常會寫入 RD = PC+4，但在這裡的 loop 用法中，rd=x0，所以簡化不處理寫回
            end
            7'b1110011: begin // ecall
                // Do nothing, just stop writing
            end
        endcase
    end
endmodule

// 5. ALU
module alu(
    input [31:0] src_a, src_b,
    input [2:0] ctrl,
    output reg [31:0] result,
    output zero,
    output reg lt // Less Than
);
    always @(*) begin
        case (ctrl)
            3'b000: result = src_a + src_b; // ADD
            3'b010: result = ($signed(src_a) < $signed(src_b)) ? 1 : 0; // SLT
            default: result = 0;
        endcase
    end
    
    assign zero = (result == 0);
    
    // 為了 B-Type 指令 (BLT)，我們需要判斷 A < B
    always @(*) begin
        lt = ($signed(src_a) < $signed(src_b));
    end
endmodule