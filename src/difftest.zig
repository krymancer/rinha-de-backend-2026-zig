//! Differential correctness harness.
//!   zig run -O ReleaseFast src/difftest.zig -- <refs.json> <val.json> [engine] [limit]
//! engine: "oracle" (brute force, default) or "knn" (index, if index.bin present)
//!
//! Loads the 3M references, then for each validation entry vectorizes the request
//! and compares our `approved` decision to the generator's `expected_approved`.
//! Asserts ZERO mismatches (E=0). Reports edge-case (score==0.6) coverage.

const std = @import("std");
const refs_mod = @import("refs.zig");
const oracle = @import("oracle.zig");
const vec = @import("vectorize.zig");
const VPAD = vec.VPAD;

fn braceEnd(buf: []const u8, start: usize) usize {
    // start points at '{'
    var i = start;
    var depth: i32 = 0;
    var in_str = false;
    while (i < buf.len) : (i += 1) {
        const c = buf[i];
        if (in_str) {
            if (c == '\\') i += 1 else if (c == '"') in_str = false;
        } else if (c == '"') {
            in_str = true;
        } else if (c == '{') {
            depth += 1;
        } else if (c == '}') {
            depth -= 1;
            if (depth == 0) return i + 1;
        }
    }
    return buf.len;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next(); // argv0
    const refs_path = it.next() orelse {
        std.debug.print("usage: difftest <refs.json> <val.json> [limit]\n", .{});
        return;
    };
    const val_path = it.next() orelse return;
    const limit: usize = if (it.next()) |l| (std.fmt.parseInt(usize, l, 10) catch std.math.maxInt(usize)) else std.math.maxInt(usize);

    const a = std.heap.page_allocator;

    const osm = @import("os.zig");
    std.debug.print("loading references {s} ...\n", .{refs_path});
    var t0 = osm.nowNs();
    var R = try refs_mod.loadJson(a, refs_path);
    defer R.deinit();
    std.debug.print("  {d} refs loaded in {d} ms\n", .{ R.n, (osm.nowNs() - t0) / 1_000_000 });

    const vbuf = try @import("os.zig").readFileAlloc(a, val_path);
    defer a.free(vbuf);

    var total: usize = 0;
    var mism: usize = 0;
    var unparseable: usize = 0;
    var edge: usize = 0;
    var edge_mism: usize = 0;

    t0 = osm.nowNs();
    var pos: usize = 0;
    while (total < limit) {
        const rs = std.mem.indexOfPos(u8, vbuf, pos, "\"request\":") orelse break;
        var p = rs + "\"request\":".len;
        while (p < vbuf.len and (vbuf[p] == ' ' or vbuf[p] == '\n')) : (p += 1) {}
        if (p >= vbuf.len or vbuf[p] != '{') break;
        const req_end = braceEnd(vbuf, p);
        const req = vbuf[p..req_end];

        // expected_approved follows the request object
        const ea_at = std.mem.indexOfPos(u8, vbuf, req_end, "\"expected_approved\":") orelse break;
        const eav = ea_at + "\"expected_approved\":".len;
        const expected = vbuf[eav] == 't';
        // expected_fraud_score (to detect edge cases == 0.6)
        const fs_at = std.mem.indexOfPos(u8, vbuf, req_end, "\"expected_fraud_score\":") orelse break;
        var fp = fs_at + "\"expected_fraud_score\":".len;
        const fstart = fp;
        while (fp < vbuf.len and vbuf[fp] != ',' and vbuf[fp] != '}') : (fp += 1) {}
        const score = std.fmt.parseFloat(f64, vbuf[fstart..fp]) catch 0;
        const is_edge = (score == 0.6);

        total += 1;
        pos = fp;

        if (is_edge) edge += 1;
        const qv = vec.vectorize(req) orelse {
            unparseable += 1;
            mism += 1;
            continue;
        };
        const res = oracle.knn(&R, &qv);
        if (res.approved != expected) {
            mism += 1;
            if (is_edge) edge_mism += 1;
            if (mism <= 10) std.debug.print("  MISMATCH #{d}: got approved={} expected={} score={d}\n", .{ mism, res.approved, expected, score });
        }
    }
    const ns = osm.nowNs() - t0;
    std.debug.print("\n=== RESULT ===\n", .{});
    std.debug.print("checked   : {d} entries in {d} ms ({d} us/query)\n", .{ total, ns / 1_000_000, if (total > 0) (ns / 1000) / total else 0 });
    std.debug.print("edge cases: {d} (score==0.6)\n", .{edge});
    std.debug.print("mismatches: {d}  (edge mismatches: {d}, unparseable: {d})\n", .{ mism, edge_mism, unparseable });
    if (mism == 0) {
        std.debug.print("PASS: E=0 on this set ✓\n", .{});
    } else {
        std.debug.print("FAIL: E={d}\n", .{mism});
        std.process.exit(1);
    }
}
