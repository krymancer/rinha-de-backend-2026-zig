//! Build-time: references.json -> index.bin (IVF k-means index, mmap-ready).
//!   indexer <references.json> <out.index.bin> [n_clusters] [iters]

const std = @import("std");
const refs_mod = @import("refs.zig");
const ivf = @import("ivf.zig");
const osm = @import("os.zig");

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    const refs_path = it.next() orelse {
        std.debug.print("usage: indexer <references.json> <out.bin> [n_clusters] [iters]\n", .{});
        return error.Args;
    };
    const out_path = it.next() orelse return error.Args;
    const n_clusters: usize = if (it.next()) |l| (std.fmt.parseInt(usize, l, 10) catch 2048) else 2048;
    const iters: usize = if (it.next()) |l| (std.fmt.parseInt(usize, l, 10) catch 12) else 12;

    const a = std.heap.page_allocator;
    var t = osm.nowNs();
    var R = try refs_mod.loadJson(a, refs_path);
    defer R.deinit();
    std.debug.print("[indexer] {d} refs in {d} ms\n", .{ R.n, (osm.nowNs() - t) / 1_000_000 });

    t = osm.nowNs();
    var idx = try ivf.build(a, &R, n_clusters, iters);
    defer idx.deinit();
    std.debug.print("[indexer] IVF: clusters={d} iters={d} blocks={d} in {d} ms\n", .{ n_clusters, iters, idx.n_blocks, (osm.nowNs() - t) / 1_000_000 });

    try ivf.save(&idx, out_path);
    std.debug.print("[indexer] wrote {s}\n", .{out_path});
}
