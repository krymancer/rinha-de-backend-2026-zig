//! Loads references.json -> int16 reference vectors (AoS, padded to 16) + labels.
//! Reference coordinates are already round4'd by the generator, so each becomes an
//! exact int16 via round(v*10000). The -1 sentinel -> -10000.

const std = @import("std");
const os = @import("os.zig");
const VPAD = @import("vectorize.zig").VPAD;

pub const Refs = struct {
    n: usize,
    vecs: []i16, // n * 16, AoS, padded; dims 14,15 = 0
    labels: []u8, // n, 1=fraud 0=legit
    alloc: std.mem.Allocator,

    pub fn deinit(self: *Refs) void {
        self.alloc.free(self.vecs);
        self.alloc.free(self.labels);
    }

    pub inline fn vec(self: *const Refs, i: usize) *const [VPAD]i16 {
        return @ptrCast(self.vecs[i * VPAD ..][0..VPAD]);
    }
};

inline fn qref(v: f64) i16 {
    return @intFromFloat(@round(v * 10000.0));
}

/// Fast forward-scan parser tailored to the fixed `[{"vector":[..14..],"label":".."},..]`
/// format. ~3M records parse in ~1-2s in ReleaseFast.
pub fn loadJson(alloc: std.mem.Allocator, path: [*:0]const u8) !Refs {
    const buf = try os.readFileAlloc(alloc, path);
    defer alloc.free(buf);

    // Count records first (one "vector": per record) to size allocations.
    var n: usize = 0;
    {
        var p: usize = 0;
        while (std.mem.indexOfPos(u8, buf, p, "\"vector\"")) |at| : (p = at + 8) n += 1;
    }
    var refs = Refs{
        .n = n,
        .vecs = try alloc.alloc(i16, n * VPAD),
        .labels = try alloc.alloc(u8, n),
        .alloc = alloc,
    };
    errdefer alloc.free(refs.vecs);
    errdefer alloc.free(refs.labels);
    @memset(refs.vecs, 0);

    var pos: usize = 0;
    var idx: usize = 0;
    while (std.mem.indexOfPos(u8, buf, pos, "\"vector\":[")) |vs| {
        var p = vs + "\"vector\":[".len;
        var d: usize = 0;
        while (d < 14) : (d += 1) {
            const start = p;
            while (p < buf.len and buf[p] != ',' and buf[p] != ']') : (p += 1) {}
            const v = std.fmt.parseFloat(f64, buf[start..p]) catch return error.BadVector;
            refs.vecs[idx * VPAD + d] = qref(v);
            if (buf[p] == ']') {
                p += 1;
                break;
            }
            p += 1; // skip comma
        }
        if (d != 13) return error.BadVectorLen;
        const ls = (std.mem.indexOfPos(u8, buf, p, "\"label\":\"") orelse return error.NoLabel) + "\"label\":\"".len;
        refs.labels[idx] = if (buf[ls] == 'f') 1 else 0;
        idx += 1;
        pos = ls;
    }
    if (idx != n) return error.CountMismatch;
    return refs;
}

test "loadJson on example-references" {
    const a = std.testing.allocator;
    // example file may not exist in all checkouts; skip gracefully.
    var refs = loadJson(a, "_official/resources/example-references.json") catch return;
    defer refs.deinit();
    try std.testing.expect(refs.n > 0);
    // first line: [0.01,0.0833,0.05,0.8261,0.1667,-1,-1,0.0432,0.25,0,1,0,0.2,0.0416] legit
    const v0 = refs.vec(0);
    try std.testing.expectEqual(@as(i16, 100), v0[0]); // 0.01
    try std.testing.expectEqual(@as(i16, -10000), v0[5]);
    try std.testing.expectEqual(@as(u8, 0), refs.labels[0]); // legit
}
