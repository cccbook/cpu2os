

## ccc: 請給一個 verilator 的專案範例，要包含 verilog, c++ 和 testbench

ChatGPT: https://chatgpt.com/c/673959ad-4444-8012-8588-72d6c0ce4532

以下是一個簡單的 `Verilator` 專案範例，展示如何使用 Verilog 模組、C++ 驅動程式和測試平台 (testbench)。此範例展示了一個簡單的加法器模組：

---

### 專案結構
```
verilator_project/
├── Makefile
├── adder.v
├── sim_main.cpp
├── testbench.cpp
```

---

### 1. Verilog 模組 (`adder.v`)
```verilog
module adder(
    input wire [3:0] a,
    input wire [3:0] b,
    output wire [4:0] sum
);
    assign sum = a + b;
endmodule
```

---

### 2. C++ 驅動程式 (`sim_main.cpp`)
這是 `Verilator` 的模擬驅動程式。

```cpp
#include "Vadder.h"
#include "verilated.h"

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);

    // Instantiate the adder module
    Vadder* adder = new Vadder;

    // Apply some inputs
    adder->a = 5;  // 4'b0101
    adder->b = 10; // 4'b1010
    adder->eval(); // Evaluate the model

    // Print the result
    printf("a: %d, b: %d, sum: %d\n", adder->a, adder->b, adder->sum);

    // Clean up
    delete adder;
    return 0;
}
```

---

### 3. 測試平台 (`testbench.cpp`)
如果需要更系統性的測試，可以加上多組測試資料：

```cpp
#include "Vadder.h"
#include "verilated.h"

void test_adder(int a, int b, int expected_sum) {
    Vadder adder;
    adder.a = a;
    adder.b = b;
    adder.eval();

    if (adder.sum == expected_sum) {
        printf("PASS: a=%d, b=%d, sum=%d\n", a, b, adder.sum);
    } else {
        printf("FAIL: a=%d, b=%d, expected=%d, got=%d\n", a, b, expected_sum, adder.sum);
    }
}

int main() {
    test_adder(5, 10, 15);
    test_adder(1, 2, 3);
    test_adder(7, 8, 15);
    return 0;
}
```

---

### 4. Makefile
使用 `Verilator` 編譯並執行模擬的 Makefile。

```makefile
# Verilator flags
VERILATOR = verilator
VERILATOR_FLAGS = --cc --exe --build

# Source files
VERILOG_SRC = adder.v
CPP_SRC = sim_main.cpp

# Target binary
TARGET = sim_main

all: $(TARGET)

$(TARGET): $(VERILOG_SRC) $(CPP_SRC)
	$(VERILATOR) $(VERILATOR_FLAGS) $(CPP_SRC) $(VERILOG_SRC) -o $(TARGET)

clean:
	rm -rf obj_dir $(TARGET)

.PHONY: all clean
```

---

### 執行步驟
1. **安裝 Verilator**（若未安裝）：
   ```bash
   sudo apt-get install verilator
   ```

2. **編譯並執行**：
   ```bash
   make
   ./sim_main
   ```

3. **結果輸出**：
   ```
   a: 5, b: 10, sum: 15
   ```

如果需要進一步擴展，測試平台可以加入更多測試資料或自動化測試腳本。