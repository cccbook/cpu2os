//A 2-level 16-bit CLA built using 4 modules of a 1-level 4bit CLA
module sum(s,p,c);
	output [3:0] s;
	input [3:0]  p,c;
	assign s = p^c;
endmodule

module carry_prop_gen(p,g,a,b);
	output [3:0] p,g;
	input [3:0] a,b;
	assign p = a^b;
	assign g = a&b;
endmodule

module carry(c1,c2,c3,c4,p,g,c0);
	output c1,c2,c3,c4;
	input [3:0] p,g;
	input c0;
	assign c1 = g[0] | c0&p[0] ; 
	assign c2 = g[1] | g[0]&p[1] | c0&p[0]&p[1];
	assign c3 = g[2] | g[1]&p[2] | g[0]&p[1]&p[2] | c0&p[0]&p[1]&p[2];
	assign c4 = g[3] | g[2]&p[3] | g[1]&p[2]&p[3] | g[0]&p[1]&p[2]&p[3] | c0&p[0]&p[1]&p[2]&p[3];
endmodule

module carry_2(c1,c2,c3,p,g,c0);
	output c1,c2,c3;
	input [3:0] p,g;
	input c0;
	assign c1 = g[0] | c0&p[0] ; 
	assign c2 = g[1] | g[0]&p[1] | c0&p[0]&p[1];
	assign c3 = g[2] | g[1]&p[2] | g[0]&p[1]&p[2] | c0&p[0]&p[1]&p[2];
	//assign c4 = g[3] | g[2]&p[3] | g[1]&p[2]&p[3] | g[0]&p[1]&p[2]&p[3] | c0&p[0]&p[1]&p[2]&p[3];
endmodule

module cla_4bit(sum,ps,gs,a,b,c0);
	output [3:0] sum;
	output gs,ps;
	//wire c4;
	input [3:0] a,b;
	input c0;
	wire [3:0] p,g;
	wire c1,c2,c3;
	carry_prop_gen cpg1(.p(p),.g(g),.a(a),.b(b));
	assign gs = g[3] | g[2]&p[3] | g[1]&p[2]&p[3] | g[0]&p[1]&p[2]&p[3]; 
	assign ps = p[3]&p[2]&p[1]&p[0];
	carry_2 ca1(.c1(c1),.c2(c2),.c3(c3),.p(p),.g(g),.c0(c0));
	sum s1(.s(sum),.p(p),.c({c3,c2,c1,c0}));
endmodule

module cla_16bit(sum,c16,a,b,c0);
	output [15:0] sum;
	output c16;
	input [15:0] a,b;
	wire [3:0] gs,ps;
	input c0;
	wire c4,c8,c12;
	cla_4bit c4b1(.sum(sum[3:0]),.ps(ps[0]),.gs(gs[0]),.a(a[3:0]),.b(b[3:0]),.c0(c0));
	cla_4bit c4b2(.sum(sum[7:4]),.ps(ps[1]),.gs(gs[1]),.a(a[7:4]),.b(b[7:4]),.c0(c4));
	cla_4bit c4b3(.sum(sum[11:8]),.ps(ps[2]),.gs(gs[2]),.a(a[11:8]),.b(b[11:8]),.c0(c8));
	cla_4bit c4b4(.sum(sum[15:12]),.ps(ps[3]),.gs(gs[3]),.a(a[15:12]),.b(b[15:12]),.c0(c12));
	carry c1(.c1(c4),.c2(c8),.c3(c12),.c4(c16),.p(ps),.g(gs),.c0(c0));
endmodule

module tb_cla_16bit();
	reg [15:0] a,b;
	reg c0;
	wire [15:0] d;
	wire cout;
	cla_16bit uut(.sum(d),.c16(cout),.a(a),.b(b),.c0(c0));
	
	initial
	begin
	#00 a=16'd10; b=16'd10; c0 = 1'b0; //10-2
	#20 a=16'd30; b=16'd20; c0 = 1'b0;
	#20 a=16'd150; b=16'd130; c0 = 1'b0; //3-5
	#20 a=16'd20000; b=16'd25555; c0 = 1'b0;
	#20 $stop;
	end
	
	initial
	begin
	$monitor("time=%3d, a=%16d, b=%16d, co=%b, d=%16d, cout=%1b",$time,a,b,c0,d,cout);
	end
		
	initial
	begin
	$dumpfile("cla_16bit.vcd");
	$dumpvars;
	end
	
endmodule
