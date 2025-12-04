import re

class RISCVAssembler:
    def __init__(self):
        # 暫存器對照表
        self.regs = {
            'zero': 0, 'ra': 1, 'sp': 2, 'gp': 3, 'tp': 4, 't0': 5, 't1': 6, 't2': 7,
            's0': 8, 's1': 9, 'a0': 10, 'a1': 11, 'a2': 12, 'a3': 13, 'a4': 14, 'a5': 15,
            'a6': 16, 'a7': 17, 's2': 18, 's3': 19, 's4': 20, 's5': 21, 's6': 22, 's7': 23,
            's8': 24, 's9': 25, 's10': 26, 's11': 27, 't3': 28, 't4': 29, 't5': 30, 't6': 31
        }
        # 加入 x0-x31 的別名
        for i in range(32):
            self.regs[f'x{i}'] = i

        self.labels = {}  # 儲存標籤位置 {label: address}
        self.code = []    # 儲存清理後的原始碼
        self.machine_code = [] # 儲存生成的機器碼

    def get_reg(self, reg_name):
        """解析暫存器名稱並回傳編號 (0-31)"""
        reg_name = reg_name.strip(',')
        if reg_name in self.regs:
            return self.regs[reg_name]
        raise ValueError(f"Unknown register: {reg_name}")

    def to_bin(self, val, bits):
        """將整數轉為二補數二進位字串"""
        val = int(val)
        if val < 0:
            val = (1 << bits) + val
        return f"{val:0{bits}b}"

    def parse_instruction(self, line):
        """解析指令行，移除註解並分割"""
        line = line.split('#')[0].strip() # 移除註解
        if not line: return None
        # 將逗號替換為空格，然後分割
        parts = re.split(r'[,\s]+', line)
        return [p for p in parts if p]

    # --- 編碼 Helper Functions ---
    
    def encode_r_type(self, opcode, funct3, funct7, rd, rs1, rs2):
        # Format: funct7 | rs2 | rs1 | funct3 | rd | opcode
        return (int(funct7, 2) << 25) | (rs2 << 20) | (rs1 << 15) | (int(funct3, 2) << 12) | (rd << 7) | int(opcode, 2)

    def encode_i_type(self, opcode, funct3, rd, rs1, imm):
        # Format: imm[11:0] | rs1 | funct3 | rd | opcode
        imm_val = int(imm)
        if imm_val < 0: imm_val += (1 << 12) # Handle negative
        imm_val &= 0xFFF # Ensure 12 bits
        return (imm_val << 20) | (rs1 << 15) | (int(funct3, 2) << 12) | (rd << 7) | int(opcode, 2)

    def encode_s_type(self, opcode, funct3, rs1, rs2, imm):
        # Format: imm[11:5] | rs2 | rs1 | funct3 | imm[4:0] | opcode
        imm_val = int(imm)
        if imm_val < 0: imm_val += (1 << 12)
        imm_11_5 = (imm_val >> 5) & 0x7F
        imm_4_0 = imm_val & 0x1F
        return (imm_11_5 << 25) | (rs2 << 20) | (rs1 << 15) | (int(funct3, 2) << 12) | (imm_4_0 << 7) | int(opcode, 2)

    def encode_b_type(self, opcode, funct3, rs1, rs2, offset):
        # SB-Type scramble: imm[12]|imm[10:5]|rs2|rs1|funct3|imm[4:1]|imm[11]|opcode
        val = int(offset)
        if val < 0: val += (1 << 13) # 13 bits (signed)
        
        imm_12 = (val >> 12) & 1
        imm_11 = (val >> 11) & 1
        imm_10_5 = (val >> 5) & 0x3F
        imm_4_1 = (val >> 1) & 0xF
        
        return (imm_12 << 31) | (imm_10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
               (int(funct3, 2) << 12) | (imm_4_1 << 8) | (imm_11 << 7) | int(opcode, 2)

    def encode_j_type(self, opcode, rd, offset):
        # UJ-Type scramble: imm[20]|imm[10:1]|imm[11]|imm[19:12]|rd|opcode
        val = int(offset)
        if val < 0: val += (1 << 21) # 21 bits (signed)

        imm_20 = (val >> 20) & 1
        imm_10_1 = (val >> 1) & 0x3FF
        imm_11 = (val >> 11) & 1
        imm_19_12 = (val >> 12) & 0xFF

        return (imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) | \
               (imm_19_12 << 12) | (rd << 7) | int(opcode, 2)

    def assemble(self, asm_text):
        self.code = []
        self.labels = {}
        self.machine_code = []
        
        # --- Pass 1: 解析標籤 (Label Parsing) ---
        pc = 0
        raw_lines = asm_text.splitlines()
        
        cleaned_lines = []
        
        for line in raw_lines:
            tokens = self.parse_instruction(line)
            if not tokens: continue
            
            # 處理 Directive (這裡只簡單忽略 .text, .globl)
            if tokens[0].startswith('.'):
                continue
                
            # 檢查是否為 Label (以 : 結尾)
            if tokens[0].endswith(':'):
                label_name = tokens[0][:-1]
                self.labels[label_name] = pc
                if len(tokens) > 1:
                    # Label 後面還有指令 (e.g., loop: add ...)
                    cleaned_lines.append({'pc': pc, 'tokens': tokens[1:]})
                    pc += 4
            else:
                cleaned_lines.append({'pc': pc, 'tokens': tokens})
                pc += 4
                
        # --- Pass 2: 生成機器碼 (Code Generation) ---
        for entry in cleaned_lines:
            pc = entry['pc']
            tokens = entry['tokens']
            mnemonic = tokens[0]
            inst_hex = 0
            
            try:
                # 處理 Pseudo-instructions
                if mnemonic == 'li': # li rd, imm -> addi rd, x0, imm
                    rd = self.get_reg(tokens[1])
                    imm = int(tokens[2])
                    inst_hex = self.encode_i_type('0010011', '000', rd, 0, imm)
                
                elif mnemonic == 'mv': # mv rd, rs -> addi rd, rs, 0
                    rd = self.get_reg(tokens[1])
                    rs = self.get_reg(tokens[2])
                    inst_hex = self.encode_i_type('0010011', '000', rd, rs, 0)

                elif mnemonic == 'j': # j label -> jal x0, offset
                    offset = self.labels[tokens[1]] - pc
                    inst_hex = self.encode_j_type('1101111', 0, offset)

                elif mnemonic == 'bgt': # bgt rs, rt, label -> blt rt, rs, label
                    rs1 = self.get_reg(tokens[1])
                    rs2 = self.get_reg(tokens[2])
                    offset = self.labels[tokens[3]] - pc
                    # 轉換為 blt (Branch Less Than)
                    inst_hex = self.encode_b_type('1100011', '100', rs2, rs1, offset) # 注意 rs1, rs2 交換

                # 處理標準指令
                elif mnemonic == 'add':
                    rd = self.get_reg(tokens[1])
                    rs1 = self.get_reg(tokens[2])
                    rs2 = self.get_reg(tokens[3])
                    inst_hex = self.encode_r_type('0110011', '000', '0000000', rd, rs1, rs2)

                elif mnemonic == 'addi':
                    rd = self.get_reg(tokens[1])
                    rs1 = self.get_reg(tokens[2])
                    imm = int(tokens[3])
                    inst_hex = self.encode_i_type('0010011', '000', rd, rs1, imm)

                elif mnemonic == 'ecall':
                    inst_hex = 0x00000073 # 固定機器碼

                # 這裡可以根據需要增加更多指令 (sub, beq, lw, sw 等)
                
                else:
                    print(f"Error: Unsupported instruction '{mnemonic}' at PC {pc}")
                    continue

                self.machine_code.append((pc, inst_hex, ' '.join(tokens)))

            except Exception as e:
                print(f"Assembly Error at line {pc}: {e}")

        return self.machine_code

    def print_output(self):
        print(f"{'PC':<8} {'Machine Code':<15} {'Assembly'}")
        print("-" * 40)
        for pc, code, asm in self.machine_code:
            print(f"0x{pc:04x}   0x{code:08x}      {asm}")

# --- 主程式：測試上一題的 1+...+10 程式碼 ---

asm_code = """
.text
.globl _start

_start:
    li t0, 0        # Sum = 0
    li t1, 1        # i = 1
    li t2, 10       # limit = 10

loop:
    bgt t1, t2, end_loop  # if i > 10 goto end
    add t0, t0, t1        # Sum += i
    addi t1, t1, 1        # i++
    j loop                # goto loop

end_loop:
    mv a0, t0       # Result to a0
    li a7, 10       # Exit syscall
    ecall
"""

assembler = RISCVAssembler()
assembler.assemble(asm_code)
assembler.print_output()