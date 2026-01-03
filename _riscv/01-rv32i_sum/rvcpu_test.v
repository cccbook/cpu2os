`timescale 1ns / 1ps

module tb_mcu;

    reg clk;
    reg rst_n;
    wire [31:0] result_a0;

    // 實例化 MCU
    // 注意：這裡假設你的 hex 檔名為 "sum.hex"
    riscv_mcu #(.HEX_FILE("asm/sum.hex")) uut (
        .clk(clk),
        .rst_n(rst_n),
        .result_a0(result_a0)
    );

    // 產生時脈
    initial begin
        clk = 0;
        forever #5 clk = ~clk; // 10ns period
    end

    // 測試流程
    initial begin
        // 1. 初始化與重置
        $dumpfile("mcu_wave.vcd"); // 產生波形檔
        $dumpvars(0, tb_mcu);
        
        rst_n = 0;
        #20;
        rst_n = 1;

        // 2. 執行程式
        $display("Starting Simulation...");

        // 執行足夠多的週期讓迴圈跑完
        // 1+..+10 迴圈大概需要 (3指令 * 10次) + overhead ~= 40-50 cycles
        #600; 

        // 3. 檢查結果
        $display("Simulation finished.");
        $display("Final Result in a0 (x10): %d (Expected: 55)", result_a0);

        if (result_a0 == 55)
            $display("TEST PASSED!");
        else
            $display("TEST FAILED!");

        $finish;
    end
    
    // 監控每一個 Cycle 的狀態 (Optional)
    always @(posedge clk) begin
        if (rst_n) begin
            // 可以在這裡 print PC 或暫存器值來 debug
            $display("PC: %h | Inst: %h | t0(sum): %d | a0(result): %d", 
          uut.pc, uut.instruction, uut.rf.regs[5], uut.rf.regs[10]);
            // $display("PC: %h, Instruction: %h, a0: %d", uut.pc, uut.instruction, uut.result_a0);
        end
    end

endmodule