`timescale 1ns / 1ps

module tb_pipeline;

    reg clk;
    reg rst_n;
    wire [31:0] result_a0;

    // 實例化 Pipeline MCU
    // 請確保這裡的 HEX_FILE 路徑正確，例如 "asm/sum.hex" 或 "sum.hex"
    riscv_pipeline #(.HEX_FILE("asm/sum.hex")) uut (
        .clk(clk),
        .rst_n(rst_n),
        .result_a0(result_a0)
    );

    // 產生時脈 (Period = 10ns)
    initial begin
        clk = 0;
        forever #5 clk = ~clk; 
    end

    // ============================================================
    // ★★★ 除錯監控區塊 (Debug Monitor) ★★★
    // ============================================================
    always @(posedge clk) begin
        if (rst_n) begin
            // 延遲 1ns 以確保訊號穩定後再讀取
            #1;
            $display("Time: %4d | PC: %h | IF_Inst: %h | t0: %2d | t1: %2d | a0: %2d | Branch: %b", 
                     $time, 
                     uut.pc,             // 目前 PC
                     uut.inst_raw,       // IF 階段抓到的指令
                     uut.rf.regs[5],     // t0 (x5)
                     uut.rf.regs[6],     // t1 (x6)
                     uut.rf.regs[10],    // a0 (x10)
                     uut.pc_src          // 是否發生跳轉 (1=Jump/Branch Taken)
            );
        end
    end

    // 測試流程
    initial begin
        $dumpfile("pipeline_wave.vcd");
        $dumpvars(0, tb_pipeline);
        
        // 1. 重置
        rst_n = 0;
        #20; // 保持 Reset 20ns
        rst_n = 1; // 釋放 Reset
        
        $display("-----------------------------------------------------------------------");
        $display(" Starting Pipeline Simulation...");
        $display("-----------------------------------------------------------------------");

        // 2. 執行
        // 迴圈跑 10 次，每次約 4 指令，考慮 Pipeline 填充與 Flush，給予足夠時間
        #1000; 

        // 3. 檢查結果
        $display("-----------------------------------------------------------------------");
        $display(" Simulation finished.");
        $display(" Final Result in a0 (x10): %d (Expected: 55)", result_a0);
        $display("-----------------------------------------------------------------------");

        if (result_a0 == 55)
            $display(" TEST PASSED!");
        else
            $display(" TEST FAILED!");

        $finish;
    end

endmodule