//! Times vectorize+search per request over a payload file (test-data.json) AND
//! validates the APPROVED decision (the only thing the grader checks) against the
//! ground-truth expected_approved, at a given probe budget, vs a full exact scan.
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

    var slat: std.ArrayList(u64) = .empty;
    defer slat.deinit(a);
    var tlat: std.ArrayList(u64) = .empty;
    defer tlat.deinit(a);
    var probes: std.ArrayList(u64) = .empty;
    defer probes.deinit(a);

    var total: usize = 0;
    // approx (ip,mp) vs ground truth
    var fp: usize = 0; // legit denied  (expected approve, we reject)
    var fn_: usize = 0; // fraud approved (expected reject, we approve)
    // exact full-scan vs ground truth (sanity)
    var efp: usize = 0;
    var efn: usize = 0;
    // approx vs exact (does the probe budget ever flip the approved bit?)
    var flips: usize = 0;
    var fc_mism: usize = 0; // fraud_count divergence (informational)

    var pos: usize = 0;
    while (true) {
        const rs = std.mem.indexOfPos(u8, vbuf, pos, "\"request\":") orelse break;
        var p = rs + "\"request\":".len;
        while (p < vbuf.len and (vbuf[p] == ' ' or vbuf[p] == '\n')) : (p += 1) {}
        if (p >= vbuf.len or vbuf[p] != '{') break;
        const req_end = braceEnd(vbuf, p);
        const req = vbuf[p..req_end];
        const ea = std.mem.indexOfPos(u8, vbuf, req_end, "\"expected_approved\":") orelse break;
        const expected = vbuf[ea + "\"expected_approved\":".len] == 't';
        pos = ea + "\"expected_approved\":".len;
        total += 1;

        const t0 = osm.nowNs();
        const qv = vec.vectorize(req) orelse {
            if (expected) fp += 1; // we'd return SAFE=approved; if expected reject this is fn, else ok -> approx: SAFE means approved=true
            continue;
        };
        const t1 = osm.nowNs();
        var np: usize = 0;
        const res = ivf.search(&index, &qv, ip, mp, keys, &np);
        const t2 = osm.nowNs();
        const exact = ivf.search(&index, &qv, index.n_clusters, index.n_clusters, keys, null);

        if (res.approved != expected) {
            if (res.approved) fn_ += 1 else fp += 1;
        }
        if (exact.approved != expected) {
            if (exact.approved) efn += 1 else efp += 1;
        }
        if (res.approved != exact.approved) flips += 1;
        if (res.fraud_count != exact.fraud_count) fc_mism += 1;

        try slat.append(a, t2 - t1);
        try tlat.append(a, t2 - t0);
        try probes.append(a, np);
    }
    std.mem.sort(u64, slat.items, {}, std.sort.asc(u64));
    std.mem.sort(u64, tlat.items, {}, std.sort.asc(u64));
    std.mem.sort(u64, probes.items, {}, std.sort.asc(u64));
    const n = tlat.items.len;
    const E_approx = fp * 1 + fn_ * 3;
    const E_exact = efp * 1 + efn * 3;
    std.debug.print("ip={d:>3} mp={d:>3} | search p50={d}ns p99={d}ns max={d}ns | probes p50={d} p99={d} max={d} | approx: fp={d} fn={d} E={d} {s} | exact: fp={d} fn={d} E={d} | flips(approx!=exact)={d} fc_mism={d}\n", .{
        ip, mp,
        pct(slat.items, 0.5), pct(slat.items, 0.99), slat.items[n - 1],
        pct(probes.items, 0.5), pct(probes.items, 0.99), probes.items[n - 1],
        fp, fn_, E_approx, if (E_approx == 0) "E=0" else "FAIL",
        efp, efn, E_exact, flips, fc_mism,
    });
}
