const print = @import("std").debug.print;

fn total(arr: []const usize) usize {
    var result: usize = 0;

    for (arr) |elem| {
        result += elem;
    }

    return result;
}

pub fn main() void {
    const numbers = [_]usize{ 1, 2, 3, 4, 5 };
    // const slice = numbers[0..3];
    const result = total(&numbers);
    print("{}\n", .{result});
}
