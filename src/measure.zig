//! Times vectorize+search per request over a payload file (test-data.json).
//!   measure <index.bin> <test-data.json>  [init_probe] [max_probe]

const std = @import("std");
const ivf = @import("ivf.zig");
const vec = @import("vectorize.zig");
const osm = @import("os.zig");

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

fn pct(s: []const u64, p: f64) u64 {
    if (s.len == 0) return 0;
    return s[@intFromFloat(p * @as(f64, @floatFromInt(s.len - 1)))];
}

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    const idx_path = it.next() orelse return error.Args;
    const data_path = it.next() orelse return error.Args;
    const ip: usize = try std.fmt.parseInt(usize, it.next() orelse "24", 10);
    const mp: usize = try std.fmt.parseInt(usize, it.next() orelse "96", 10);

    const a = std.heap.page_allocator;
    var index = try ivf.map(a, idx_path, true);
    defer index.deinit();
    const vbuf = try osm.readFileAlloc(a, data_path);
    defer a.free(vbuf);
    const keys = try a.alloc(u64, index.n_clusters);
    defer a.free(keys);

    var vlat: std.ArrayList(u64) = .empty;
    defer vlat.deinit(a);
    var slat: std.ArrayList(u64) = .empty;
    defer slat.deinit(a);
    var tlat: std.ArrayList(u64) = .empty;
    defer tlat.deinit(a);
    var probes: std.ArrayList(u64) = .empty;
    defer probes.deinit(a);

    var pos: usize = 0;
    var sink: u64 = 0;
    var mism: usize = 0;
    while (true) {
        const rs = std.mem.indexOfPos(u8, vbuf, pos, "\"request\":") orelse break;
        var p = rs + "\"request\":".len;
        while (p < vbuf.len and (vbuf[p] == ' ' or vbuf[p] == '\n')) : (p += 1) {}
        if (p >= vbuf.len or vbuf[p] != '{') break;
        const req_end = braceEnd(vbuf, p);
        const req = vbuf[p..req_end];
        pos = req_end;

        const t0 = osm.nowNs();
        const qv = vec.vectorize(req) orelse continue;
        const t1 = osm.nowNs();
        var np: usize = 0;
        const res = ivf.search(&index, &qv, ip, mp, keys, &np);
        const t2 = osm.nowNs();
        // Exact reference: probe every cluster (no confident early-stop, no budget
        // cap) so the 5-NN are the true brute-force nearest. Any divergence in
        // fraud_count is an approximation error that would raise the grader's E.
        const exact = ivf.search(&index, &qv, index.n_clusters, index.n_clusters, keys, null);
        if (exact.fraud_count != res.fraud_count) mism += 1;
        sink = sink *% 1000003 +% res.fraud_count; // order-sensitive checksum
        try vlat.append(a, t1 - t0);
        try slat.append(a, t2 - t1);
        try tlat.append(a, t2 - t0);
        try probes.append(a, np);
    }
    std.mem.sort(u64, vlat.items, {}, std.sort.asc(u64));
    std.mem.sort(u64, slat.items, {}, std.sort.asc(u64));
    std.mem.sort(u64, tlat.items, {}, std.sort.asc(u64));
    std.mem.sort(u64, probes.items, {}, std.sort.asc(u64));
    const n = tlat.items.len;
    std.debug.print("requests={d} ip={d} mp={d} (checksum={d}) approx-vs-exact mismatches={d} -> {s}\n", .{ n, ip, mp, sink, mism, if (mism == 0) "E=0 (exact)" else "APPROX ERROR" });
    std.debug.print("vectorize ns: p50={d} p99={d} max={d}\n", .{ pct(vlat.items, 0.5), pct(vlat.items, 0.99), vlat.items[n - 1] });
    std.debug.print("search    ns: p50={d} p99={d} max={d}\n", .{ pct(slat.items, 0.5), pct(slat.items, 0.99), slat.items[n - 1] });
    std.debug.print("TOTAL     ns: p50={d} p99={d} max={d}\n", .{ pct(tlat.items, 0.5), pct(tlat.items, 0.99), tlat.items[n - 1] });
    std.debug.print("probes      : p50={d} p99={d} max={d}\n", .{ pct(probes.items, 0.5), pct(probes.items, 0.99), probes.items[n - 1] });
}
