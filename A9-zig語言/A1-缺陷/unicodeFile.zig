const std = @import("std");
const print = std.debug.print;
const mem = std.mem; // will be used to compare bytes

pub fn main() !void {
    // var file = try std.fs.cwd().openFile("test.txt", .{});
    // defer file.close();

    // var buf_reader = std.io.bufferedReader(file.reader());
    // var in_stream = buf_reader.reader();

    // var buf: [1024]u8 = undefined;
    // while (try in_stream.readUntilDelimiterOrEof(&buf, '\n')) |line| {
    //     // do something with line...
    //     print("{s}", line);
    // }
    const file = try std.fs.cwd().openFile("test2.txt", .{}); // 英文，沒問題
    // const file = try std.fs.cwd().openFile("test1.txt", .{}); // 中文，有亂碼
    defer file.close();

    var buffer: [100]u8 = undefined;
    const bytes_read = try file.readAll(&buffer);
    var text = buffer[0..bytes_read];
    print("{d}\n", .{bytes_read});
    print("{s}\n", .{text});
    return;
}
