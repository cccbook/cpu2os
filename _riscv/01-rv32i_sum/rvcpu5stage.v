`timescale 1ns / 1ps

module riscv_pipeline #(
    parameter HEX_FILE = "sum.hex"
)(
    input clk,
    input rst_n,
    output [31:0] result_a0 // 用於 Testbench 觀察 x10 (a0)
);

    // ==========================================
    // 1. IF Stage (Instruction Fetch)
    // ==========================================
    wire [31:0] pc_next, pc_plus4, pc_target;
    wire pc_src; // 來自 MEM 階段的跳轉決策訊號
    reg [31:0] pc;
    wire [31:0] inst_raw;

    // PC Register
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) pc <= 0;
        else pc <= pc_next;
    end

    // Next PC Logic
    assign pc_plus4 = pc + 4;
    assign pc_next = (pc_src) ? pc_target : pc_plus4;

    // Instruction Memory
    instruction_memory #(.FILENAME(HEX_FILE)) imem (
        .addr(pc),
        .inst(inst_raw)
    );

    // ==========================================
    // IF/ID Pipeline Register
    // ==========================================
    reg [31:0] if_id_pc;
    reg [31:0] if_id_inst;
    
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            if_id_pc <= 0;
            if_id_inst <= 0;
        end else if (pc_src) begin
            // Flush IF/ID
            if_id_pc <= 0;
            if_id_inst <= 0; 
        end else begin
            if_id_pc <= pc;
            if_id_inst <= inst_raw;
        end
    end

    // ==========================================
    // 2. ID Stage (Instruction Decode)
    // ==========================================
    wire [4:0] id_rs1 = if_id_inst[19:15];
    wire [4:0] id_rs2 = if_id_inst[24:20];
    wire [4:0] id_rd  = if_id_inst[11:7];
    wire [31:0] id_rdata1, id_rdata2;
    wire [31:0] id_imm;
    
    // Control Signals
    wire id_reg_write, id_alu_src, id_branch, id_jump;
    wire [2:0] id_alu_ctrl;

    // Register File wiring
    wire wb_reg_write;
    wire [4:0] wb_rd;
    wire [31:0] wb_wdata;

    register_file rf (
        .clk(clk), .rst_n(rst_n),
        .we(wb_reg_write),
        .raddr1(id_rs1), .raddr2(id_rs2),
        .waddr(wb_rd), .wdata(wb_wdata),
        .rdata1(id_rdata1), .rdata2(id_rdata2),
        .monitor_a0(result_a0)
    );

    imm_gen ig (.inst(if_id_inst), .imm_ext(id_imm));

    control_unit ctrl (
        .opcode(if_id_inst[6:0]),
        .funct3(if_id_inst[14:12]),
        .reg_write(id_reg_write),
        .alu_src(id_alu_src),
        .branch(id_branch),
        .jump(id_jump),
        .alu_ctrl(id_alu_ctrl)
    );

    // ==========================================
    // ID/EX Pipeline Register
    // ==========================================
    reg [31:0] id_ex_pc, id_ex_rdata1, id_ex_rdata2, id_ex_imm;
    reg [4:0]  id_ex_rs1, id_ex_rs2, id_ex_rd;
    reg [2:0]  id_ex_alu_ctrl;
    reg        id_ex_reg_write, id_ex_alu_src, id_ex_branch, id_ex_jump;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            id_ex_pc <= 0; id_ex_rdata1 <= 0; id_ex_rdata2 <= 0; id_ex_imm <= 0;
            id_ex_rs1 <= 0; id_ex_rs2 <= 0; id_ex_rd <= 0;
            id_ex_reg_write <= 0; id_ex_branch <= 0; id_ex_jump <= 0;
        end else if (pc_src) begin
            // Flush ID/EX
            id_ex_reg_write <= 0; id_ex_branch <= 0; id_ex_jump <= 0;
            id_ex_rd <= 0; 
        end else begin
            id_ex_pc <= if_id_pc;
            id_ex_rdata1 <= id_rdata1;
            id_ex_rdata2 <= id_rdata2;
            id_ex_imm <= id_imm;
            id_ex_rs1 <= id_rs1;
            id_ex_rs2 <= id_rs2;
            id_ex_rd <= id_rd;
            // Control Signals
            id_ex_reg_write <= id_reg_write;
            id_ex_alu_src <= id_alu_src;
            id_ex_branch <= id_branch;
            id_ex_jump <= id_jump;
            id_ex_alu_ctrl <= id_alu_ctrl;
        end
    end

    // ==========================================
    // 3. EX Stage (Execute)
    // ==========================================
    wire [31:0] alu_in_a, alu_in_b_final;
    wire [31:0] fwd_data_a, fwd_data_b;
    wire [1:0]  forward_a, forward_b;
    wire [31:0] ex_alu_result;
    wire ex_zero, ex_lt;

    assign fwd_data_a = (forward_a == 2'b10) ? ex_mem_alu_result :
                        (forward_a == 2'b01) ? wb_wdata : id_ex_rdata1;
    
    assign fwd_data_b = (forward_b == 2'b10) ? ex_mem_alu_result :
                        (forward_b == 2'b01) ? wb_wdata : id_ex_rdata2;

    assign alu_in_a = fwd_data_a;
    assign alu_in_b_final = (id_ex_alu_src) ? id_ex_imm : fwd_data_b;

    alu main_alu (
        .src_a(alu_in_a),
        .src_b(alu_in_b_final),
        .ctrl(id_ex_alu_ctrl),
        .result(ex_alu_result),
        .zero(ex_zero),
        .lt(ex_lt)
    );

    wire [31:0] ex_pc_target = id_ex_pc + id_ex_imm;

    // ==========================================
    // EX/MEM Pipeline Register
    // ==========================================
    reg [31:0] ex_mem_alu_result, ex_mem_pc_target;
    reg [4:0]  ex_mem_rd;
    reg        ex_mem_reg_write, ex_mem_branch, ex_mem_jump;
    reg        ex_mem_zero, ex_mem_lt;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            ex_mem_alu_result <= 0; ex_mem_pc_target <= 0; ex_mem_rd <= 0;
            ex_mem_reg_write <= 0; ex_mem_branch <= 0; ex_mem_jump <= 0;
            ex_mem_zero <= 0; ex_mem_lt <= 0;
        end else if (pc_src) begin
            // Flush EX/MEM: 跳轉發生，清除當前 EX 階段指令
            ex_mem_reg_write <= 0;
            ex_mem_branch <= 0;
            ex_mem_jump <= 0;
            ex_mem_rd <= 0;
            ex_mem_alu_result <= 0; 
            ex_mem_pc_target <= 0;
        end else begin
            ex_mem_reg_write <= id_ex_reg_write;
            ex_mem_branch <= id_ex_branch;
            ex_mem_jump <= id_ex_jump;
            ex_mem_alu_result <= ex_alu_result;
            ex_mem_pc_target <= ex_pc_target;
            ex_mem_rd <= id_ex_rd;
            ex_mem_zero <= ex_zero;
            ex_mem_lt <= ex_lt;
        end
    end

    // ==========================================
    // 4. MEM Stage (Memory Access)
    // ==========================================
    assign pc_src = (ex_mem_branch & ex_mem_lt) | ex_mem_jump; 
    assign pc_target = ex_mem_pc_target;

    // ==========================================
    // MEM/WB Pipeline Register
    // ==========================================
    reg [31:0] mem_wb_alu_result;
    reg [4:0]  mem_wb_rd;
    reg        mem_wb_reg_write;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            mem_wb_alu_result <= 0; mem_wb_rd <= 0; mem_wb_reg_write <= 0;
        end else begin
            mem_wb_alu_result <= ex_mem_alu_result;
            mem_wb_rd <= ex_mem_rd;
            mem_wb_reg_write <= ex_mem_reg_write;
        end
    end

    // ==========================================
    // 5. WB Stage (Write Back)
    // ==========================================
    assign wb_reg_write = mem_wb_reg_write;
    assign wb_rd = mem_wb_rd;
    assign wb_wdata = mem_wb_alu_result;

    // ==========================================
    // Forwarding Unit
    // ==========================================
    forwarding_unit fwd_unit (
        .rs1(id_ex_rs1),
        .rs2(id_ex_rs2),
        .ex_mem_rd(ex_mem_rd),
        .mem_wb_rd(mem_wb_rd),
        .ex_mem_reg_write(ex_mem_reg_write),
        .mem_wb_reg_write(mem_wb_reg_write),
        .forward_a(forward_a),
        .forward_b(forward_b)
    );

endmodule


// ==========================================
// 子模組定義
// ==========================================

module forwarding_unit(
    input [4:0] rs1, rs2, ex_mem_rd, mem_wb_rd,
    input ex_mem_reg_write, mem_wb_reg_write,
    output reg [1:0] forward_a, forward_b
);
    always @(*) begin
        // Forward A
        if (ex_mem_reg_write && (ex_mem_rd != 0) && (ex_mem_rd == rs1))
            forward_a = 2'b10;
        else if (mem_wb_reg_write && (mem_wb_rd != 0) && (mem_wb_rd == rs1))
            forward_a = 2'b01;
        else
            forward_a = 2'b00;

        // Forward B
        if (ex_mem_reg_write && (ex_mem_rd != 0) && (ex_mem_rd == rs2))
            forward_b = 2'b10;
        else if (mem_wb_reg_write && (mem_wb_rd != 0) && (mem_wb_rd == rs2))
            forward_b = 2'b01;
        else
            forward_b = 2'b00;
    end
endmodule

module instruction_memory #(parameter FILENAME = "sum.hex")(input [31:0] addr, output [31:0] inst);
    reg [31:0] mem [0:63]; 
    integer i; // [修正] 將變數宣告移出 initial block
    
    initial begin
        for (i=0; i<64; i=i+1) mem[i] = 0;
        $readmemh(FILENAME, mem);
    end
    assign inst = mem[addr[31:2]]; 
endmodule

module register_file(
    input clk, rst_n, we,
    input [4:0] raddr1, raddr2, waddr,
    input [31:0] wdata,
    output [31:0] rdata1, rdata2,
    output [31:0] monitor_a0
);
    reg [31:0] regs [0:31];
    integer i;

    // 寫入邏輯 (循序電路)
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            for (i=0; i<32; i=i+1) regs[i] <= 0;
        end else if (we && waddr != 0) begin
            regs[waddr] <= wdata;
        end
    end

    // 讀取邏輯 (修正後：支援 WB -> ID 的內部 Forwarding)
    // 如果讀取地址 == 寫入地址 且 正在寫入 (we=1)，直接輸出 wdata
    assign rdata1 = (raddr1 == 0) ? 0 :
                    (we && raddr1 == waddr) ? wdata : regs[raddr1];

    assign rdata2 = (raddr2 == 0) ? 0 :
                    (we && raddr2 == waddr) ? wdata : regs[raddr2];

    assign monitor_a0 = regs[10];
endmodule

module imm_gen(input [31:0] inst, output reg [31:0] imm_ext);
    always @(*) begin
        case (inst[6:0])
            7'b0010011: imm_ext = {{20{inst[31]}}, inst[31:20]}; // ADDI
            7'b1100011: imm_ext = {{19{inst[31]}}, inst[31], inst[7], inst[30:25], inst[11:8], 1'b0}; // Branch
            7'b1101111: imm_ext = {{11{inst[31]}}, inst[31], inst[19:12], inst[20], inst[30:21], 1'b0}; // JAL
            default:    imm_ext = 0;
        endcase
    end
endmodule

module control_unit(
    input [6:0] opcode, input [2:0] funct3,
    output reg reg_write, alu_src, branch, jump,
    output reg [2:0] alu_ctrl
);
    always @(*) begin
        reg_write = 0; alu_src = 0; branch = 0; jump = 0; alu_ctrl = 0;
        case (opcode)
            7'b0110011: begin reg_write = 1; alu_ctrl = 3'b000; end // add
            7'b0010011: begin reg_write = 1; alu_src = 1; alu_ctrl = 3'b000; end // addi
            7'b1100011: begin branch = 1; alu_ctrl = 3'b010; end // blt
            7'b1101111: begin jump = 1; end // jal
        endcase
    end
endmodule

module alu(
    input [31:0] src_a, src_b, input [2:0] ctrl,
    output reg [31:0] result, output zero, output reg lt
);
    always @(*) begin
        case (ctrl)
            3'b000: result = src_a + src_b;
            3'b010: result = ($signed(src_a) < $signed(src_b)) ? 1 : 0;
            default: result = 0;
        endcase
    end
    assign zero = (result == 0);
    always @(*) lt = ($signed(src_a) < $signed(src_b));
endmodule