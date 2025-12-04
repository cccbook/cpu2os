# 編譯
iverilog -o rvcpu5stage_test rvcpu5stage.v rvcpu5stage_test.v

# 執行
vvp rvcpu5stage_test
