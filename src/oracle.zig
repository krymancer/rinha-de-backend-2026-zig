//! Brute-force exact KNN over int16 references. This is the CORRECTNESS ORACLE:
//! it is the simplest faithful reimplementation of data-generator/main.c's
//! knn_classify (k=5, squared euclidean, ties broken by lowest original index),
//! and every fast search path is differential-tested against it.

const std = @import("std");
const Refs = @import("refs.zig").Refs;
const VPAD = @import("vectorize.zig").VPAD;

pub const K = 5;
pub const IDX_BITS = 22; // n < 3M < 2^22
pub const IDX_MASK: u64 = (1 << IDX_BITS) - 1;

/// squared euclidean distance over 16 padded dims (dims 14,15 are 0 on both sides).
pub inline fn sqdist(a: *const [VPAD]i16, b: *const [VPAD]i16) i64 {
    var s: i64 = 0;
    inline for (0..VPAD) |d| {
        const diff: i64 = @as(i64, a[d]) - @as(i64, b[d]);
        s += diff * diff;
    }
    return s;
}

/// Pack (distance, original index) so ascending u64 order == (dist asc, index asc),
/// exactly replicating the generator's first-seen-wins tie-break.
pub inline fn packKey(dist: i64, idx: usize) u64 {
    return (@as(u64, @intCast(dist)) << IDX_BITS) | @as(u64, idx);
}

pub const Result = struct { fraud_count: u8, approved: bool };

/// fraud_count among the 5 nearest; approved = fraud_count < 3 (score < 0.6).
pub fn knn(refs: *const Refs, query: *const [VPAD]i16) Result {
    var best: [K]u64 = .{std.math.maxInt(u64)} ** K;
    var worst_i: usize = 0; // index in `best` holding the current maximum key
    var worst_key: u64 = std.math.maxInt(u64);

    var i: usize = 0;
    while (i < refs.n) : (i += 1) {
        const d = sqdist(query, refs.vec(i));
        const key = packKey(d, i);
        if (key < worst_key) {
            best[worst_i] = key;
            // recompute worst
            worst_key = best[0];
            worst_i = 0;
            inline for (1..K) |j| {
                if (best[j] > worst_key) {
                    worst_key = best[j];
                    worst_i = j;
                }
            }
        }
    }

    var fraud: u8 = 0;
    inline for (0..K) |j| {
        const ri = best[j] & IDX_MASK;
        fraud += refs.labels[ri];
    }
    return .{ .fraud_count = fraud, .approved = fraud < 3 };
}

test "knn tie-break by lowest index" {
    const a = std.testing.allocator;
    var refs = Refs{
        .n = 6,
        .vecs = try a.alloc(i16, 6 * VPAD),
        .labels = try a.alloc(u8, 6),
        .alloc = a,
    };
    defer refs.deinit();
    @memset(refs.vecs, 0);
    // dim0 distances^2: 100,100,400,900,2500,100  ; labels L,F,F,F,F,F
    const d0 = [_]i16{ 10, 10, 20, 30, 50, 10 };
    const lab = [_]u8{ 0, 1, 1, 1, 1, 1 };
    for (0..6) |k| {
        refs.vecs[k * VPAD] = d0[k];
        refs.labels[k] = lab[k];
    }
    var query: [VPAD]i16 = .{0} ** VPAD;
    const r = knn(&refs, &query);
    // 5 nearest by (dist,idx): (100,0)L,(100,1)F,(100,5)F,(400,2)F,(900,3)F => 4 frauds
    try std.testing.expectEqual(@as(u8, 4), r.fraud_count);
    try std.testing.expectEqual(false, r.approved);
    _ = &query;
}
