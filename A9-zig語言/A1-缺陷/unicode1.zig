const print = @import("std").debug.print;
const mem = @import("std").mem; // will be used to compare bytes

pub fn main() void {
    const bytes = "hello";
    print("{}\n", .{@TypeOf(bytes)}); // *const [5:0]u8
    print("{d}\n", .{bytes.len}); // 5
    print("{c}\n", .{bytes[1]}); // 'e'
    print("{d}\n", .{bytes[5]}); // 0
    print("Hello\n", .{});
    print("中文\n", .{});
    print("中文是否支援得好呢？\n", .{});
    print("{s}是否支援得好呢？\n", .{"中文"});
}
