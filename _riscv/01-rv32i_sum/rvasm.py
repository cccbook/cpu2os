import re
import argparse
import sys

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
        self.machine_code = [] # 儲存生成的機器碼 [(pc, hex_code, asm_source)]

    def get_reg(self, reg_name):
        """解析暫存器名稱並回傳編號 (0-31)"""
        reg_name = reg_name.strip(',').strip()
        if reg_name in self.regs:
            return self.regs[reg_name]
        raise ValueError(f"Unknown register: {reg_name}")

    def parse_instruction(self, line):
        """解析指令行，移除註解並分割"""
        line = line.split('#')[0].strip() # 移除註解
        if not line: return None
        # 將逗號替換為空格，然後分割
        parts = re.split(r'[,\s]+', line)
        return [p for p in parts if p]

    # --- 編碼 Helper Functions (Bitwise Operations) ---
    def encode_r_type(self, opcode, funct3, funct7, rd, rs1, rs2):
        return (int(funct7, 2) << 25) | (rs2 << 20) | (rs1 << 15) | (int(funct3, 2) << 12) | (rd << 7) | int(opcode, 2)

    def encode_i_type(self, opcode, funct3, rd, rs1, imm):
        imm_val = int(imm)
        if imm_val < 0: imm_val += (1 << 12)
        imm_val &= 0xFFF
        return (imm_val << 20) | (rs1 << 15) | (int(funct3, 2) << 12) | (rd << 7) | int(opcode, 2)

    def encode_b_type(self, opcode, funct3, rs1, rs2, offset):
        val = int(offset)
        if val < 0: val += (1 << 13)
        
        imm_12 = (val >> 12) & 1
        imm_11 = (val >> 11) & 1
        imm_10_5 = (val >> 5) & 0x3F
        imm_4_1 = (val >> 1) & 0xF
        
        return (imm_12 << 31) | (imm_10_5 << 25) | (rs2 << 20) | (rs1 << 15) | \
               (int(funct3, 2) << 12) | (imm_4_1 << 8) | (imm_11 << 7) | int(opcode, 2)

    def encode_j_type(self, opcode, rd, offset):
        val = int(offset)
        if val < 0: val += (1 << 21)

        imm_20 = (val >> 20) & 1
        imm_10_1 = (val >> 1) & 0x3FF
        imm_11 = (val >> 11) & 1
        imm_19_12 = (val >> 12) & 0xFF

        return (imm_20 << 31) | (imm_10_1 << 21) | (imm_11 << 20) | \
               (imm_19_12 << 12) | (rd << 7) | int(opcode, 2)

    def assemble(self, asm_text):
        """主組譯邏輯"""
        self.labels = {}
        self.machine_code = []
        
        # --- Pass 1: 解析標籤 ---
        pc = 0
        raw_lines = asm_text.splitlines()
        cleaned_lines = []
        
        for line in raw_lines:
            tokens = self.parse_instruction(line)
            if not tokens: continue
            if tokens[0].startswith('.'): continue # Ignore directives
                
            if tokens[0].endswith(':'):
                label_name = tokens[0][:-1]
                self.labels[label_name] = pc
                if len(tokens) > 1:
                    cleaned_lines.append({'pc': pc, 'tokens': tokens[1:]})
                    pc += 4
            else:
                cleaned_lines.append({'pc': pc, 'tokens': tokens})
                pc += 4
                
        # --- Pass 2: 生成機器碼 ---
        for entry in cleaned_lines:
            pc = entry['pc']
            tokens = entry['tokens']
            mnemonic = tokens[0]
            inst_hex = 0
            
            try:
                # Pseudo-instructions
                if mnemonic == 'li':
                    rd = self.get_reg(tokens[1])
                    imm = int(tokens[2])
                    inst_hex = self.encode_i_type('0010011', '000', rd, 0, imm)
                
                elif mnemonic == 'mv':
                    rd = self.get_reg(tokens[1])
                    rs = self.get_reg(tokens[2])
                    inst_hex = self.encode_i_type('0010011', '000', rd, rs, 0)

                elif mnemonic == 'j':
                    offset = self.labels[tokens[1]] - pc
                    inst_hex = self.encode_j_type('1101111', 0, offset)

                elif mnemonic == 'bgt': # bgt rs, rt, label -> blt rt, rs, label
                    rs1 = self.get_reg(tokens[1])
                    rs2 = self.get_reg(tokens[2])
                    offset = self.labels[tokens[3]] - pc
                    inst_hex = self.encode_b_type('1100011', '100', rs2, rs1, offset)

                # Standard Instructions
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
                    inst_hex = 0x00000073

                else:
                    raise ValueError(f"Unsupported instruction: {mnemonic}")

                self.machine_code.append((pc, inst_hex, ' '.join(tokens)))

            except Exception as e:
                print(f"[Error] Line at PC {pc}: {e}")
                sys.exit(1)

    def get_output_string(self):
        """生成格式化的輸出字串"""
        output = []
        output.append(f"{'PC':<8} {'Hex Code':<12} {'Assembly'}")
        output.append("-" * 40)
        for pc, code, asm in self.machine_code:
            output.append(f"0x{pc:04x}   0x{code:08x}   {asm}")
        return "\n".join(output)

    def get_hex_only_string(self):
        """僅生成 Hex 內容 (適合 Verilog readmemh)"""
        return "\n".join([f"{code:08x}" for _, code, _ in self.machine_code])

# --- Main Entry Point ---

def main():
    # 設定參數解析器
    parser = argparse.ArgumentParser(description="Simple RISC-V RV32I Assembler")
    
    # 定義輸入與輸出參數
    parser.add_argument("input_file", help="Path to the input assembly file (.asm or .s)")
    parser.add_argument("output_file", help="Path to the output file")
    
    # 選擇性參數：是否只輸出純 Hex
    parser.add_argument("--hex-only", action="store_true", help="Output only hex codes (no PC or source text)")

    args = parser.parse_args()

    # 讀取輸入檔
    try:
        with open(args.input_file, 'r', encoding='utf-8') as f:
            asm_content = f.read()
    except FileNotFoundError:
        print(f"Error: Input file '{args.input_file}' not found.")
        sys.exit(1)

    # 執行組譯
    assembler = RISCVAssembler()
    print(f"Assembling {args.input_file}...")
    assembler.assemble(asm_content)

    # 準備輸出內容
    if args.hex_only:
        result_content = assembler.get_hex_only_string()
    else:
        result_content = assembler.get_output_string()

    # 寫入輸出檔
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            f.write(result_content)
        print(f"Success! Output written to '{args.output_file}'")
    except Exception as e:
        print(f"Error writing to file: {e}")

if __name__ == "__main__":
    main()