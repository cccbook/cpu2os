const expect = @import("std").testing.expect;

test "optional" {
    var found_index: ?usize = null;
    const data = [_]i32{ 1, 2, 3, 4, 5, 6, 7, 8, 12 };
    for (data, 0..) |v, i| {
        if (v == 10) found_index = i;
    }
    try expect(found_index == null);
}

test "orelse" {
    var a: ?f32 = null;
    var b = a orelse 0;
    try expect(b == 0);
    try expect(@TypeOf(b) == f32);
}

test "orelse unreachable" {
    const a: ?f32 = 5;
    const b = a orelse unreachable;
    const c = a.?; // .? is a shorthand for orelse unreachable
    try expect(b == c);
    try expect(@TypeOf(c) == f32);
}

test "if optional payload capture" {
    const a: ?i32 = 5;
    if (a != null) {
        const value = a.?; // .? is a shorthand for orelse unreachable
        _ = value;
    }

    var b: ?i32 = 5;
    if (b) |*value| {
        value.* += 1;
    }
    try expect(b.? == 6); 
}
