//! Build the index over the real 3M references, then validate E=0 AND measure
//! per-query latency + probe stats over a validation set.
//!   zig run -O ReleaseFast src/bench.zig -- <refs.json> <val.json> [leaf_max]

const std = @import("std");
const refs_mod = @import("refs.zig");
const index = @import("index.zig");
const vec = @import("vectorize.zig");
const osm = @import("os.zig");
const VPAD = vec.VPAD;

fn braceEnd(buf: []const u8, start: usize) usize {
    var i = start;
    var depth: i32 = 0;
    var in_str = false;
    while (i < buf.len) : (i += 1) {
        const c = buf[i];
        if (in_str) {
            if (c == '\\') i += 1 else if (c == '"') in_str = false;
        } else if (c == '"') in_str = true else if (c == '{') depth += 1 else if (c == '}') {
            depth -= 1;
            if (depth == 0) return i + 1;
        }
    }
    return buf.len;
}

fn pct(sorted: []const u64, p: f64) u64 {
    if (sorted.len == 0) return 0;
    const i: usize = @intFromFloat(p * @as(f64, @floatFromInt(sorted.len - 1)));
    return sorted[i];
}

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    const refs_path = it.next() orelse {
        std.debug.print("usage: bench <refs.json> <val.json> [leaf_max]\n", .{});
        return;
    };
    const val_path = it.next() orelse return;
    const leaf_max: usize = if (it.next()) |l| (std.fmt.parseInt(usize, l, 10) catch 1024) else 1024;
    const cap: usize = if (it.next()) |l| (std.fmt.parseInt(usize, l, 10) catch 1_000_000) else 1_000_000;

    const a = std.heap.page_allocator;

    var t = osm.nowNs();
    var R = try refs_mod.loadJson(a, refs_path);
    defer R.deinit();
    std.debug.print("refs: {d} loaded in {d} ms\n", .{ R.n, (osm.nowNs() - t) / 1_000_000 });

    t = osm.nowNs();
    var idx = try index.build(a, &R, leaf_max);
    defer idx.deinit();
    std.debug.print("index: leaf_max={d} nodes={d} built in {d} ms\n", .{ leaf_max, idx.n_nodes, (osm.nowNs() - t) / 1_000_000 });

    const vbuf = try osm.readFileAlloc(a, val_path);
    defer a.free(vbuf);

    var lat: std.ArrayList(u64) = .empty;
    defer lat.deinit(a);
    var probes_all: std.ArrayList(u64) = .empty;
    defer probes_all.deinit(a);

    var total: usize = 0;
    var mism: usize = 0;
    var edge: usize = 0;
    var edge_mism: usize = 0;
    var probe_sum: u64 = 0;

    var pos: usize = 0;
    while (true) {
        const rs = std.mem.indexOfPos(u8, vbuf, pos, "\"request\":") orelse break;
        var p = rs + "\"request\":".len;
        while (p < vbuf.len and (vbuf[p] == ' ' or vbuf[p] == '\n')) : (p += 1) {}
        if (p >= vbuf.len or vbuf[p] != '{') break;
        const req_end = braceEnd(vbuf, p);
        const req = vbuf[p..req_end];
        const ea_at = std.mem.indexOfPos(u8, vbuf, req_end, "\"expected_approved\":") orelse break;
        const expected = vbuf[ea_at + "\"expected_approved\":".len] == 't';
        const fs_at = std.mem.indexOfPos(u8, vbuf, req_end, "\"expected_fraud_score\":") orelse break;
        var fp = fs_at + "\"expected_fraud_score\":".len;
        const fstart = fp;
        while (fp < vbuf.len and vbuf[fp] != ',' and vbuf[fp] != '}') : (fp += 1) {}
        const score = std.fmt.parseFloat(f64, vbuf[fstart..fp]) catch 0;
        const is_edge = (score == 0.6);
        pos = fp;
        total += 1;
        if (is_edge) edge += 1;

        const t0 = osm.nowNs();
        const qv = vec.vectorize(req) orelse {
            mism += 1;
            continue;
        };
        var nprobe: usize = 0;
        const res = index.searchCore(&idx, &qv, &nprobe, cap);
        const dt = osm.nowNs() - t0;
        try lat.append(a, dt);
        try probes_all.append(a, nprobe);
        probe_sum += nprobe;
        if (res.approved != expected) {
            mism += 1;
            if (is_edge) edge_mism += 1;
            if (mism <= 8) std.debug.print("  MISMATCH: got={} exp={} score={d} fraud={d}\n", .{ res.approved, expected, score, res.fraud_count });
        }
    }

    std.mem.sort(u64, lat.items, {}, std.sort.asc(u64));
    std.mem.sort(u64, probes_all.items, {}, std.sort.asc(u64));
    std.debug.print("\n=== {d} queries (leaf_max={d}, nodes={d}) ===\n", .{ total, leaf_max, idx.n_nodes });
    std.debug.print("latency (vectorize+search) ns:  p50={d}  p90={d}  p99={d}  p99.9={d}  max={d}\n", .{ pct(lat.items, 0.5), pct(lat.items, 0.9), pct(lat.items, 0.99), pct(lat.items, 0.999), lat.items[lat.items.len - 1] });
    std.debug.print("latency us:                     p50={d:.2}  p99={d:.2}  max={d:.2}\n", .{ @as(f64, @floatFromInt(pct(lat.items, 0.5))) / 1000.0, @as(f64, @floatFromInt(pct(lat.items, 0.99))) / 1000.0, @as(f64, @floatFromInt(lat.items[lat.items.len - 1])) / 1000.0 });
    std.debug.print("leaves visited:  mean={d:.1}  p50={d}  p99={d}  max={d}\n", .{ @as(f64, @floatFromInt(probe_sum)) / @as(f64, @floatFromInt(total)), pct(probes_all.items, 0.5), pct(probes_all.items, 0.99), probes_all.items[probes_all.items.len - 1] });
    std.debug.print("edge cases: {d}   mismatches: {d} (edge: {d})\n", .{ edge, mism, edge_mism });
    std.debug.print("{s}\n", .{if (mism == 0) "PASS E=0 ✓" else "FAIL E>0"});
    if (mism != 0) std.process.exit(1);
}
