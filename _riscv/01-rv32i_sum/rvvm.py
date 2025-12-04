import sys

class RISCV_VM:
    def __init__(self):
        # 32 個通用暫存器 x0-x31，初始化為 0
        self.regs = [0] * 32
        
        # Program Counter (PC)
        self.pc = 0
        
        # 指令記憶體 (Key: Address, Value: Instruction Integer)
        self.memory = {}
        
        # 程式結束旗標
        self.running = True

    def load_hex(self, hex_file):
        """讀取 hex 檔案並載入模擬記憶體"""
        try:
            with open(hex_file, 'r') as f:
                lines = f.readlines()
                
            addr = 0
            print(f"Loading {hex_file}...")
            for line in lines:
                line = line.strip()
                if not line: continue
                # 將 hex 字串轉為整數 (例如 "00100313" -> 1049363)
                inst = int(line, 16)
                self.memory[addr] = inst
                addr += 4 # RISC-V 指令長度為 4 bytes
            
            print(f"Loaded {len(self.memory)} instructions.")
            
        except FileNotFoundError:
            print(f"Error: File {hex_file} not found.")
            sys.exit(1)

    def to_signed(self, val, bits):
        """處理二補數，將無號整數轉為有號整數"""
        if val & (1 << (bits - 1)):
            return val - (1 << bits)
        return val

    def dump_regs(self):
        """列印目前暫存器狀態"""
        print("-" * 60)
        print(f"PC: 0x{self.pc:04x}")
        for i in range(0, 32, 4):
            r_str = ""
            for j in range(4):
                reg_idx = i + j
                reg_name = f"x{reg_idx}"
                # 標記常用的 ABI 名稱
                if reg_idx == 10: reg_name += "(a0)"
                if reg_idx == 5:  reg_name += "(t0)"
                if reg_idx == 6:  reg_name += "(t1)"
                r_str += f"{reg_name:>7}: {self.regs[reg_idx]:<10}"
            print(r_str)
        print("-" * 60)

    def run(self):
        print("\nStarting execution...")
        instruction_count = 0
        
        while self.running:
            # --- 1. Fetch (取指) ---
            if self.pc not in self.memory:
                print(f"Error: PC 0x{self.pc:04x} out of bounds or invalid.")
                break
                
            inst = self.memory[self.pc]

            print(f"PC: 0x{self.pc:04x} | Inst: 0x{inst:08x} | t0: {self.regs[5]}")

            instruction_count += 1
            
            # --- 2. Decode (解碼) ---
            opcode = inst & 0x7F
            rd     = (inst >> 7) & 0x1F
            funct3 = (inst >> 12) & 0x7
            rs1    = (inst >> 15) & 0x1F
            rs2    = (inst >> 20) & 0x1F
            funct7 = (inst >> 25) & 0x7F
            
            # --- 3. Execute (執行) ---
            
            # R-Type: ADD (Opcode: 0110011)
            if opcode == 0x33:
                if funct3 == 0x0 and funct7 == 0x00: # add
                    self.regs[rd] = self.regs[rs1] + self.regs[rs2]
                    # 模擬 32-bit overflow 行為
                    self.regs[rd] &= 0xFFFFFFFF 
                else:
                    print(f"Unknown R-Type funct3/7 at PC {self.pc:04x}")

            # I-Type: ADDI, ECALL (Opcode: 0010011, 1110011)
            elif opcode == 0x13: # addi
                imm_i = self.to_signed((inst >> 20) & 0xFFF, 12)
                if funct3 == 0x0: # addi
                    self.regs[rd] = self.regs[rs1] + imm_i
                    self.regs[rd] &= 0xFFFFFFFF
            
            elif opcode == 0x73: # ecall
                print("\n[System] Ecall triggered.")
                if self.regs[17] == 10: # a7=10 (Exit)
                    print("[System] Program exited normally.")
                    self.running = False
                else:
                    print(f"[System] Unknown syscall {self.regs[17]}")
                    self.running = False
            
            # B-Type: BLT (Opcode: 1100011)
            # 你的組譯器將 bgt 轉為 blt (less than)
            elif opcode == 0x63: 
                # Decode B-Type Immediate
                imm_12 = (inst >> 31) & 1
                imm_10_5 = (inst >> 25) & 0x3F
                imm_4_1 = (inst >> 8) & 0xF
                imm_11 = (inst >> 7) & 1
                
                imm_val = (imm_12 << 12) | (imm_11 << 11) | (imm_10_5 << 5) | (imm_4_1 << 1)
                offset = self.to_signed(imm_val, 13)

                taken = False
                val1 = self.to_signed(self.regs[rs1], 32)
                val2 = self.to_signed(self.regs[rs2], 32)

                if funct3 == 0x0: # beq
                    taken = (val1 == val2)
                elif funct3 == 0x1: # bne
                    taken = (val1 != val2)
                elif funct3 == 0x4: # blt
                    taken = (val1 < val2)
                elif funct3 == 0x5: # bge
                    taken = (val1 >= val2)
                
                if taken:
                    self.pc += offset
                    self.regs[0] = 0 # 確保 x0 永遠為 0
                    continue # 跳過最後的 PC+4 更新

            # J-Type: JAL (Opcode: 1101111)
            elif opcode == 0x6F: 
                # Decode J-Type Immediate
                imm_20 = (inst >> 31) & 1
                imm_10_1 = (inst >> 21) & 0x3FF
                imm_11 = (inst >> 20) & 1
                imm_19_12 = (inst >> 12) & 0xFF
                
                imm_val = (imm_20 << 20) | (imm_19_12 << 12) | (imm_11 << 11) | (imm_10_1 << 1)
                offset = self.to_signed(imm_val, 21)
                
                # JAL: rd = PC + 4 (Link), PC = PC + offset
                if rd != 0:
                    self.regs[rd] = self.pc + 4
                
                self.pc += offset
                self.regs[0] = 0
                continue # 跳過最後的 PC+4 更新

            else:
                print(f"Unknown Opcode 0x{opcode:x} at PC {self.pc:04x}")
                self.running = False

            # --- 4. Update State ---
            self.regs[0] = 0 # Hardwire x0 to 0
            self.pc += 4 # 前進到下一個指令

        print(f"\nExecution finished in {instruction_count} cycles.")
        print(f"Final Result (a0/x10): {self.regs[10]}")

if __name__ == "__main__":
    vm = RISCV_VM()
    # 預設讀取 sum.hex，你也可以改成 sys.argv[1]
    input_file = "sum.hex" 
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
        
    vm.load_hex(input_file)
    vm.run()
    vm.dump_regs()