//! Build-time: references.json -> index.bin (mmap-ready kd-tree).
//!   indexer <references.json> <out.index.bin> [leaf_max]

const std = @import("std");
const refs_mod = @import("refs.zig");
const index = @import("index.zig");
const osm = @import("os.zig");

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    const refs_path = it.next() orelse {
        std.debug.print("usage: indexer <references.json> <out.bin> [leaf_max]\n", .{});
        return error.Args;
    };
    const out_path = it.next() orelse return error.Args;
    const leaf_max: usize = if (it.next()) |l| (std.fmt.parseInt(usize, l, 10) catch 64) else 64;

    const a = std.heap.page_allocator;
    var t = osm.nowNs();
    var R = try refs_mod.loadJson(a, refs_path);
    defer R.deinit();
    std.debug.print("[indexer] {d} refs in {d} ms\n", .{ R.n, (osm.nowNs() - t) / 1_000_000 });

    t = osm.nowNs();
    var idx = try index.build(a, &R, leaf_max);
    defer idx.deinit();
    std.debug.print("[indexer] built kd-tree: nodes={d} blocks={d} leaf_max={d} in {d} ms\n", .{ idx.n_nodes, idx.n_blocks, leaf_max, (osm.nowNs() - t) / 1_000_000 });

    try index.saveIndex(&idx, out_path);
    std.debug.print("[indexer] wrote {s}\n", .{out_path});
}
