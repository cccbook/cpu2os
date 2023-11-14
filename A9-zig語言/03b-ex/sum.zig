const std = @import("std");

fn sum(n: usize) usize {
    var result: usize = 0;
    var i: usize = 1;

    while (i <= n) : (i += 1) {
        result += i;
    }

    return result;
}

pub fn main() void {
    const n: usize = 5;
    const result = sum(n);
    std.debug.print("{}\n", .{result});
}
