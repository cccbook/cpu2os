/* Generated by Yosys 0.47 (git sha1 647d61dd9, clang++ 16.0.0 -fPIC -O3) */

(* top =  1  *)
(* src = "adder.v:2.1-8.10" *)
module adder(a, b, sum);
  wire _00_;
  wire _01_;
  wire _02_;
  wire _03_;
  wire _04_;
  wire _05_;
  wire _06_;
  wire _07_;
  wire _08_;
  wire _09_;
  wire _10_;
  wire _11_;
  wire _12_;
  wire _13_;
  wire _14_;
  (* src = "adder.v:3.17-3.18" *)
  input [3:0] a;
  wire [3:0] a;
  (* src = "adder.v:4.17-4.18" *)
  input [3:0] b;
  wire [3:0] b;
  (* src = "adder.v:5.18-5.21" *)
  output [4:0] sum;
  wire [4:0] sum;
  assign _00_ = a[3] & b[3];
  assign _01_ = a[3] ^ b[3];
  assign _02_ = ~(a[2] & b[2]);
  assign _03_ = _01_ & ~(_02_);
  assign _04_ = _03_ | _00_;
  assign _05_ = ~(a[2] ^ b[2]);
  assign _06_ = _01_ & ~(_05_);
  assign _07_ = ~(a[1] & b[1]);
  assign _08_ = a[1] ^ b[1];
  assign _09_ = a[0] & b[0];
  assign _10_ = _09_ & _08_;
  assign _11_ = _07_ & ~(_10_);
  assign _12_ = _06_ & ~(_11_);
  assign sum[4] = _12_ | _04_;
  assign sum[1] = _09_ ^ _08_;
  assign sum[2] = _11_ ^ _05_;
  assign _13_ = ~(_11_ | _05_);
  assign _14_ = _13_ | ~(_02_);
  assign sum[3] = _14_ ^ _01_;
  assign sum[0] = a[0] ^ b[0];
endmodule