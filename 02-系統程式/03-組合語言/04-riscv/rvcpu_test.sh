# 編譯
iverilog -o rvcpu_test rvcpu.v rvcpu_test.v

# 執行
vvp rvcpu_test