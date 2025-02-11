`define PC   R[15]   // 祘Α璸计竟
`define LR   R[14]   // 硈挡既竟
`define SP   R[13]   // 帮舼既竟
`define SW   R[12]   // 篈既竟
// 篈既竟篨夹じ
`define N    `SW[31] // 璽腹篨夹
`define Z    `SW[30] // 箂篨夹
`define C    `SW[29] // 秈篨夹
`define V    `SW[28] // 犯篨夹
`define I    `SW[7]  // 祑砰い耞砛
`define T    `SW[6]  // 硁砰い耞砛
`define M    `SW[0]  // 家Αじ

module cpu0c(input clock); // CPU0-Mini еcpu0mc 家舱
  parameter [7:0] LD=8'h00,ST=8'h01,LDB=8'h02,STB=8'h03,LDR=8'h04,STR=8'h05,
    LBR=8'h06,SBR=8'h07,ADDI=8'h08,CMP=8'h10,MOV=8'h12,ADD=8'h13,SUB=8'h14,
    MUL=8'h15,DIV=8'h16,AND=8'h18,OR=8'h19,XOR=8'h1A,ROL=8'h1C,ROR=8'h1D,
    SHL=8'h1E,SHR=8'h1F,JEQ=8'h20,JNE=8'h21,JLT=8'h22,JGT=8'h23,JLE=8'h24,
    JGE=8'h25,JMP=8'h26,SWI=8'h2A,CALL=8'h2B,RET=8'h2C,IRET=8'h2D,
    PUSH=8'h30,POP=8'h31,PUSHB=8'h32,POPB=8'h33;
  reg signed [31:0] R [0:15];   // 既竟 R[0..15] 单 16  32 じ既竟
  reg signed [31:0] IR;         // 既竟 IR
  reg [7:0] m [0:128];          // ず场е癘拘砰
  reg [7:0] op;                 // 跑计笲衡絏 op
  reg [3:0] ra, rb, rc;         // 跑计既竟腹 ra, rb, rc
  reg [4:0] c5;                 // 跑计5 じ盽计 c5
  reg signed [11:0] c12;        // 跑计12 じ盽计 c12
  reg signed [15:0] c16;        // 跑计16 じ盽计 c16
  reg signed [23:0] c24;        // 跑计24 じ盽计 c24
  reg signed [31:0] sp, jaddr, laddr, raddr;
  reg signed [31:0] temp;

  initial  // ﹍て
  begin
    `PC = 0;                    // 盢 PC 砞癬笆 0
    R[0] = 0;                   // 盢 R[0] 既竟眏砞﹚ 0
    {m[0],m[1],m[2],m[3]}    = 32'h001F0018; // 0000       LD   R1, K1
    {m[4],m[5],m[6],m[7]}    = 32'h002F0010; // 0004       LD   R2, K0
    {m[8],m[9],m[10],m[11]}  = 32'h003F0014; // 0008       LD   R3, SUM
    {m[12],m[13],m[14],m[15]}= 32'h13221000; // 000C LOOP: ADD  R2, R2, R1
    {m[16],m[17],m[18],m[19]}= 32'h13332000; // 0010       ADD  R3, R3, R2
    {m[20],m[21],m[22],m[23]}= 32'h26FFFFF4; // 0014       JMP  LOOP
    {m[24],m[25],m[26],m[27]}= 32'h00000000; // 0018 K0:   WORD 0
    {m[28],m[29],m[30],m[31]}= 32'h00000001; // 001C K1:   WORD 1
    {m[32],m[33],m[34],m[35]}= 32'h00000000; // 0020 SUM:  WORD 0
  end
  
  always @(posedge clock) begin //  clock タ娩絫牟祇
      IR = {m[`PC], m[`PC+1], m[`PC+2], m[`PC+3]};  // 耝顶琿IR=m[PC], 4  Byte 癘拘砰
      `PC = `PC+4;                                  // 耝ЧΘPC 玡秈
      {op,ra,rb,rc,c12} = IR;                      // 秆絏顶琿盢 IR 秆 {op, ra, rb, rc, c12}
      c5  = IR[4:0];
      c24 = IR[23:0];
      c16 = IR[15:0];
      jaddr = `PC+c16;
	  laddr = R[rb]+c16;
	  raddr = R[rb]+R[rc];
//	  $display(" ra=%d, rb=%d, rc=%d, jaddr=%x, laddr=%x, raddr=%x", ra, rb, rc, jaddr, laddr, raddr);
      case (op) // 沮 OP 磅︽癸莱笆
        LD: begin   // 更 R[ra] = m[addr]
          R[ra] = {m[laddr], m[laddr+1], m[laddr+2], m[laddr+3]};
          $write("%4dns %8x : LD  %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        ST: begin   // 纗 m[addr] = R[ra]
          {m[laddr], m[laddr+1], m[laddr+2], m[laddr+3]} = R[ra];
          $write("%4dns %8x : ST  %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        LDB:begin   // 更byte;     LDB Ra, [Rb+ Cx];   Ra<=(byte)[Rb+ Cx]
          R[ra] = { 24'b0, m[laddr] };
          $write("%4dns %8x : LDB %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        STB:begin   // 纗byte;     STB Ra, [Rb+ Cx];   Ra=>(byte)[Rb+ Cx]
          m[laddr] = R[ra][7:0];
          $write("%4dns %8x : STB %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        LDR:begin   // LD  Rc ;  LDR Ra, [Rb+Rc];    Ra<=[Rb+ Rc]
          R[ra] = {m[raddr], m[raddr+1], m[raddr+2], m[raddr+3]};
          $write("%4dns %8x : LDR %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        STR:begin   // ST  Rc ;  STR Ra, [Rb+Rc];    Ra=>[Rb+ Rc]
          {m[raddr], m[raddr+1], m[raddr+2], m[raddr+3]} = R[ra];
          $write("%4dns %8x : STR %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        LBR:begin   // LDB  Rc ; LBR Ra, [Rb+Rc];    Ra<=(byte)[Rb+ Rc]
          R[ra] = { 24'b0, m[raddr] };
          $write("%4dns %8x : LBR %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        SBR:begin   // STB  Rc ; SBR Ra, [Rb+Rc];    Ra=>(byte)[Rb+ Rc]
          m[raddr] = R[ra][7:0];
          $write("%4dns %8x : SBR %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        MOV:begin   // 簿笆;        MOV Ra, Rb;         Ra<=Rb
		  R[ra] = R[rb];
          $write("%4dns %8x : MOV %x,%x", $stime, `PC, ra, rb);
          end
        CMP:begin   // ゑ耕;        CMP Ra, Rb;         SW=(Ra >=< Rb)
		  temp = R[ra]-R[rb];
		  `N=(temp<0);`Z=(temp==0);
          $write("%4dns %8x : CMP %x,%x; SW=%x", $stime, `PC, ra, rb, `SW);
          end
        ADDI:begin  // R[a] = Rb+c16;  // ミ猭;   LDI Ra, Rb+Cx; Ra<=Rb + Cx
		  R[ra] = R[rb]+c16;
          $write("%4dns %8x : ADDI %x,%x,%-4x", $stime, `PC, ra, rb, c16);
          end
        ADD: begin  // 猭 R[ra] = R[rb]+R[rc]
          R[ra] = R[rb]+R[rc];
          $write("%4dns %8x : ADD %x,%x,%-4x", $stime, `PC, ra, rb, rc);
          end
        SUB:begin   // 搭猭;        SUB Ra, Rb, Rc;     Ra<=Rb-Rc
          R[ra] = R[rb]-R[rc];
          $write("%4dns %8x : SUB %x,%x,%-4x", $stime, `PC, ra, rb, rc);
          end
        MUL:begin   // 猭;        MUL Ra, Rb, Rc;     Ra<=Rb*Rc
          R[ra] = R[rb]*R[rc];
          $write("%4dns %8x : MUL %x,%x,%-4x", $stime, `PC, ra, rb, rc);
          end
        DIV:begin   // 埃猭;        DIV Ra, Rb, Rc;     Ra<=Rb/Rc
          R[ra] = R[rb]/R[rc];
          $write("%4dns %8x : DIV %x,%x,%-4x", $stime, `PC, ra, rb, rc);
          end
        AND:begin   // じ AND;    AND Ra, Rb, Rc;     Ra<=Rb and Rc
          R[ra] = R[rb]&R[rc];
          $write("%4dns %8x : AND %x,%x,%-4x", $stime, `PC, ra, rb, rc);
          end
        OR:begin    // じ OR;     OR Ra, Rb, Rc;         Ra<=Rb or Rc
          R[ra] = R[rb]|R[rc];
          $write("%4dns %8x : OR  %x,%x,%-4x", $stime, `PC, ra, rb, rc);
          end
        XOR:begin   // じ XOR;    XOR Ra, Rb, Rc;     Ra<=Rb xor Rc
          R[ra] = R[rb]^R[rc];
          $write("%4dns %8x : XOR %x,%x,%-4x", $stime, `PC, ra, rb, rc);
          end
        SHL:begin   // オ簿;    SHL Ra, Rb, Cx;     Ra<=Rb << Cx
          R[ra] = R[rb]<<c5;
          $write("%4dns %8x : SHL %x,%x,%-4x", $stime, `PC, ra, rb, c5);
          end
        SHR:begin   // 簿;        SHR Ra, Rb, Cx;     Ra<=Rb >> Cx
          R[ra] = R[rb]+R[rc];
          $write("%4dns %8x : SHR %x,%x,%-4x", $stime, `PC, ra, rb, c5);
          end		  
        JMP:begin   // 铬臘 PC = PC + cx24
          `PC = `PC + c24;
          $write("%4dns %8x : JMP %-8x", $stime, `PC, c24);
          end
        JEQ:begin   // 铬臘 (单);        JEQ Cx;        if SW(=) PC  PC+Cx
		  if (`Z) `PC=`PC+c24;
          $write("%4dns %8x : JEQ %-8x", $stime, `PC, c24);
          end
        JNE:begin   // 铬臘 (ぃ单);    JNE Cx;     if SW(!=) PC  PC+Cx
		  if (!`Z) `PC=`PC+c24;
          $write("%4dns %8x : JNE %-8x", $stime, `PC, c24);
          end
        JLT:begin   // 铬臘 ( < );        JLT Cx;     if SW(<) PC  PC+Cx
          if (`N) `PC=`PC+c24;
          $write("%4dns %8x : JLT %-8x", $stime, `PC, c24);
          end
        JGT:begin   // 铬臘 ( > );        JGT Cx;     if SW(>) PC  PC+Cx
          if (!`N&&!`Z) `PC=`PC+c24;
          $write("%4dns %8x : JGT %-8x", $stime, `PC, c24);
          end
        JLE:begin   // 铬臘 ( <= );        JLE Cx;     if SW(<=) PC  PC+Cx  
          if (`N || `Z) `PC=`PC+c24;
          $write("%4dns %8x : JLE %-8x", $stime, `PC, c24);
          end
        JGE:begin   // 铬臘 ( >= );        JGE Cx;     if SW(>=) PC  PC+Cx
          if (!`N || `Z) `PC=`PC+c24;
          $write("%4dns %8x : JGE %-8x", $stime, `PC, c24);
          end
        SWI:begin   // 硁い耞;    SWI Cx;         LR <= PC; PC <= Cx; INT<=1
          `LR=`PC;`PC= c24; `I = 1'b1;
          $write("%4dns %8x : SWI %-8x", $stime, `PC, c24);
          end
        CALL:begin  // 铬捌祘Α;    CALL Cx;     LR<=PC; PC<=PC+Cx
          `LR=`PC;`PC=`PC + c24;
          $write("%4dns %8x : CALL %-8x", $stime, `PC, c24);
          end
        RET:begin   // ;            RET;         PC <= LR
          `PC=`LR;
          $write("%4dns %8x : RET, PC=%x", $stime, `PC);
          end
        IRET:begin  // い耞;        IRET;         PC <= LR; INT<=0
          `PC=`LR;`I = 1'b0;
          $write("%4dns %8x : RET, PC=%x", $stime, `PC);
          end
        PUSH:begin  // 崩 word;    PUSH Ra;    SP-=4;[SP]<=Ra;
          sp = `SP-4; `SP = sp; {m[sp], m[sp+1], m[sp+2], m[sp+3]} = R[ra];
          $write("%4dns %8x : PUSH %-x", $stime, `PC, R[ra]);
		  end
        POP:begin   // 紆 word;    POP Ra;     Ra=[SP];SP+=4;
          sp = `SP+4; `SP = sp; R[ra]={m[sp], m[sp+1], m[sp+2], m[sp+3]};
          $write("%4dns %8x : POP %-x", $stime, `PC, R[ra]);
          end
        PUSHB:begin // 崩 byte;    PUSHB Ra;   SP--;[SP]<=Ra;(byte)
          sp = `SP-1; `SP = sp; m[sp] = R[ra];
          $write("%4dns %8x : PUSHB %-x", $stime, `PC, R[ra]);
          end
        POPB:begin  // 紆 byte;    POPB Ra;  Ra<=[SP];SP++;(byte)
          sp = `SP+1; `SP = sp; R[ra]=m[sp];
          $write("%4dns %8x : POPB %-x", $stime, `PC, R[ra]);
          end
      endcase
      $display(" R[%2d]=%4d", ra, R[ra]); // 陪ボヘ夹既竟
  end
endmodule

module main;                // 代刚祘Α秨﹍
reg clock;                  //  clock 跑计

cpu0c cpu(clock);          //  cpu0mc 矪瞶竟

initial clock = 0;          // 秨﹍ clock 砞﹚ 0
always #10 clock=~clock;    // –筳 10 ー盢 clock は玻ネ秅戳 20 ー
initial #640 $finish;       //  640 ー氨ゎ代刚(硂 R[1] 琌 1+2+...+10=55 挡狦)
endmodule
