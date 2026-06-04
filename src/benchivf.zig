//! IVF tuning harness:
//!   benchivf <refs.json> <val.json> <n_clusters> <iters> <init_probe> <max_probe>

const std = @import("std");
const refs_mod = @import("refs.zig");
const ivf = @import("ivf.zig");
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

fn pct(s: []const u64, p: f64) u64 {
    if (s.len == 0) return 0;
    return s[@intFromFloat(p * @as(f64, @floatFromInt(s.len - 1)))];
}

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    const refs_path = it.next() orelse return error.Args;
    const val_path = it.next() orelse return error.Args;
    const ncl: usize = try std.fmt.parseInt(usize, it.next() orelse "2048", 10);
    const iters: usize = try std.fmt.parseInt(usize, it.next() orelse "12", 10);
    const ip: usize = try std.fmt.parseInt(usize, it.next() orelse "12", 10);
    const mp: usize = try std.fmt.parseInt(usize, it.next() orelse "64", 10);

    const a = std.heap.page_allocator;
    var t = osm.nowNs();
    var R = try refs_mod.loadJson(a, refs_path);
    defer R.deinit();
    std.debug.print("refs {d} in {d}ms\n", .{ R.n, (osm.nowNs() - t) / 1_000_000 });

    t = osm.nowNs();
    var index = try ivf.build(a, &R, ncl, iters);
    defer index.deinit();
    std.debug.print("ivf: clusters={d} iters={d} blocks={d} built {d}ms\n", .{ ncl, iters, index.n_blocks, (osm.nowNs() - t) / 1_000_000 });

    const vbuf = try osm.readFileAlloc(a, val_path);
    defer a.free(vbuf);
    const keys = try a.alloc(u64, ncl);
    defer a.free(keys);

    var lat: std.ArrayList(u64) = .empty;
    defer lat.deinit(a);
    var probes: std.ArrayList(u64) = .empty;
    defer probes.deinit(a);
    var total: usize = 0;
    var mism: usize = 0;
    var edge: usize = 0;
    var edge_mism: usize = 0;
    var psum: u64 = 0;

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
        const fa = std.mem.indexOfPos(u8, vbuf, req_end, "\"expected_fraud_score\":") orelse break;
        var fp = fa + "\"expected_fraud_score\":".len;
        const fs = fp;
        while (fp < vbuf.len and vbuf[fp] != ',' and vbuf[fp] != '}') : (fp += 1) {}
        const score = std.fmt.parseFloat(f64, vbuf[fs..fp]) catch 0;
        const is_edge = (score == 0.6);
        pos = fp;
        total += 1;
        if (is_edge) edge += 1;
        const t0 = osm.nowNs();
        const qv = vec.vectorize(req) orelse {
            mism += 1;
            continue;
        };
        var np: usize = 0;
        const res = ivf.search(&index, &qv, ip, mp, keys, &np);
        const dt = osm.nowNs() - t0;
        try lat.append(a, dt);
        try probes.append(a, np);
        psum += np;
        if (res.approved != expected) {
            mism += 1;
            if (is_edge) edge_mism += 1;
        }
    }
    std.mem.sort(u64, lat.items, {}, std.sort.asc(u64));
    std.mem.sort(u64, probes.items, {}, std.sort.asc(u64));
    std.debug.print("latency us: p50={d:.1} p90={d:.1} p99={d:.1} max={d:.1}\n", .{
        @as(f64, @floatFromInt(pct(lat.items, 0.5))) / 1000,
        @as(f64, @floatFromInt(pct(lat.items, 0.9))) / 1000,
        @as(f64, @floatFromInt(pct(lat.items, 0.99))) / 1000,
        @as(f64, @floatFromInt(lat.items[lat.items.len - 1])) / 1000,
    });
    std.debug.print("probes: mean={d:.1} p50={d} p99={d} max={d}\n", .{ @as(f64, @floatFromInt(psum)) / @as(f64, @floatFromInt(total)), pct(probes.items, 0.5), pct(probes.items, 0.99), probes.items[probes.items.len - 1] });
    std.debug.print("edge={d} mismatches={d} (edge {d}) -> {s}\n", .{ edge, mism, edge_mism, if (mism == 0) "PASS E=0" else "FAIL" });
    if (mism != 0) std.process.exit(1);
}
