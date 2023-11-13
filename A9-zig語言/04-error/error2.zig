const expect = @import("std").testing.expect;

// test "out of bounds" {
//     const a = [3]u8{ 1, 2, 3 };
//     var index: u8 = 5;
//     const b = a[index];
//     _ = b;
// }

fn arrayGet() error{OutOfBounds}!i32 {
    const a = [3]u8{ 1, 2, 3 };
    var index: u8 = 5;
    const b = a[index];
    // _ = b;
    return b;
}

test "out of bounds" {
    var r = arrayGet()!i32 catch |err| {
        try expect(err == error.Oops);
        return 0;
    };
    try expect(r==error.OutOfBounds);
}
