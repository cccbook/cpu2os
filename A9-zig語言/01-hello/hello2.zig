const print = @import("std").debug.print;

pub fn main() void {
    print("Hello, {s}!\n", .{"World"});
}

