// https://medium.com/@eddo2626/lets-learn-zig-4-using-c-libraries-in-zig-5fcc3206f0dc

const std = @import("std");

const c = @cImport({
    @cInclude("stdio.h");
    @cInclude("stdlib.h");
});

pub fn main() !void {
    var ret = c.printf("hello from c world!\n");
    std.debug.print("C call return value: {d}\n", .{ret});

    const buf = c.malloc(10);
    if (buf == null) {
        std.debug.print("ERROR while allocating memory!\n", .{});
        return;
    }
    std.debug.print("buf address: {any}\n", .{buf});
    c.free(buf);
}
