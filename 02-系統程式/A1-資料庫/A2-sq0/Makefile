# ====================================================================
# Makefile for sq0 Mini-SQLite Engine
# ====================================================================

# 1. 變數定義 (Variables)

# C 編譯器
CC = gcc

# C 編譯器旗標：
# -Wall: 開啟所有常見警告
# -Wextra: 開啟額外警告
# -std=c99: 使用 C99 標準
# -g: 包含偵錯資訊
CFLAGS = -Wall -Wextra -std=c99 -g

# 專案名稱/輸出執行檔名稱
TARGET = sq0_db

# 核心原始碼檔案 (.c)
SRCS = main.c sq0.c pager.c btree.c parser.c vm.c util.c lexer.c

# 測試檔案
TEST_SRC = test_sq0.c
TEST_BIN = run_tests
TEST_DB_FILE = test_sq0.db

# 所有目標檔案 (.o)
OBJS = $(SRCS:.c=.o)
# 核心函式庫目標檔案 (排除 main.o)
LIB_OBJS = $(filter-out main.o, $(OBJS))
# 測試目標檔案 (.o)
TEST_OBJ = $(TEST_SRC:.c=.o)


# 2. 預設目標 (Default Target)

# 執行 'make' 時會編譯主程式
.PHONY: all
all: $(TARGET)


# 3. 主執行檔目標 (Target Executable)

# 鏈接所有核心 .o 檔案來創建 sq0_db
$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(OBJS) -o $(TARGET)


# 4. 測試目標 (Test Targets)

# 執行 'make test' 時會編譯並運行測試
.PHONY: test
test: $(TEST_BIN)
	@echo "--- Running Tests ---"
	./$(TEST_BIN)

# 鏈接核心 .o 檔案和測試 .o 檔案來創建 run_tests
$(TEST_BIN): $(LIB_OBJS) $(TEST_OBJ)
	$(CC) $(CFLAGS) $(LIB_OBJS) $(TEST_OBJ) -o $(TEST_BIN)

# 5. 編譯規則 (Compilation Rule)

# 如何從 .c 檔案創建 .o 檔案
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# 6. 檔案依賴關係 (Dependencies)

# 確保標頭檔變更時，相應的 .c 檔案會被重新編譯
main.o: main.c sq0.h btree.h
pager.o: pager.c sq0.h
btree.o: btree.c sq0.h pager.h util.h
parser.o: parser.c sq0.h
vm.o: vm.c sq0.h btree.h util.h
util.o: util.c sq0.h
lexer.o: lexer.c lexer.h # 雖然目前 parser 沒用 lexer，但保持依賴完整性
test_sq0.o: test_sq0.c sq0.h btree.h vm.h

# 7. 清理目標 (Clean Target)

# 移除所有生成的檔案：主程式、測試程式、中間目標檔案、和測試數據庫
.PHONY: clean
clean:
	rm -f $(TARGET) $(OBJS) $(TEST_BIN) $(TEST_OBJ) $(TEST_DB_FILE)