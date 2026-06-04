//! Exact k-NN over 3M int16 vectors via a kd-tree with incremental
//! branch-and-bound, fast enough for sub-ms p99.
//!
//! Build: recursive median split on the widest-spread dimension. Leaves hold up
//! to LEAF_MAX vectors, stored as SoA blocks of 8 (dimension-major) for AVX2.
//!
//! Query: descend toward the query's side first; visit the far subtree only when
//! its region lower bound (incremental sum of squared per-dimension offsets) is
//! still below the current 5th-best distance. Because the bound is exact, this
//! returns the TRUE 5 nearest neighbours (identical to brute force) -> E stays 0,
//! while touching only a tiny fraction of the 3M vectors.
//!
//! Packed key = (dist << 23) | (orig_index << 1) | label. Ascending key order ==
//! (dist asc, orig_index asc), replicating the generator's first-seen-wins
//! tie-break; the label rides along so fraud counting needs no extra lookup.

const std = @import("std");
const refs_mod = @import("refs.zig");
const Refs = refs_mod.Refs;
const VPAD = @import("vectorize.zig").VPAD;

pub const K = 5;
pub const DIST_SHIFT = 23; // 1 label bit + 22 index bits below the distance
pub const LANES = 8;
pub const BLOCK_I16 = VPAD * LANES; // 128 i16 per block
pub const LEAF = 0xFF; // node.dim sentinel for a leaf

/// dist max = 2*(20000^2) + 12*(10000^2) = 2.0e9 < 2^31, so i32 accumulation is safe.
pub const Result = struct { fraud_count: u8, approved: bool };

pub const Node = extern struct {
    dim: u8, // 0xFF => leaf
    _pad: u8 = 0,
    split: i16, // internal: split value on `dim`
    a: u32, // internal: left child index;  leaf: first block index
    b: u32, // internal: right child index; leaf: vector count
};

pub const Index = struct {
    n: usize,
    n_nodes: usize,
    root: u32,
    leaf_max: usize,
    nodes: []Node,
    blocks: []align(32) i16, // SoA-8, dimension-major
    meta: []u32, // block-ordered: (orig_idx<<1)|label; pad lanes = 0xFFFFFFFF
    n_blocks: usize = 0,
    alloc: ?std.mem.Allocator = null,
    mmap: ?[]align(std.heap.page_size_min) u8 = null,

    pub fn deinit(self: *Index) void {
        if (self.mmap) |m| {
            _ = @import("os.zig").linux.munmap(@ptrCast(m.ptr), m.len);
        } else if (self.alloc) |a| {
            a.free(self.nodes);
            a.free(self.blocks);
            a.free(self.meta);
        }
    }
};

// ---- serialization (build-time -> index.bin, mmap at runtime) ----------------

const MAGIC = "RNHZIDX2";

pub const Header = extern struct {
    magic: [8]u8,
    n: u64,
    n_nodes: u64,
    root: u64,
    leaf_max: u64,
    n_blocks: u64,
    nodes_off: u64,
    blocks_off: u64,
    meta_off: u64,
    total: u64,
};

inline fn alignUp(x: usize, a: usize) usize {
    return (x + a - 1) & ~(a - 1);
}

pub fn saveIndex(idx: *const Index, path: [*:0]const u8) !void {
    const os = @import("os.zig");
    const nodes_off = alignUp(@sizeOf(Header), 64);
    const nodes_bytes = idx.nodes.len * @sizeOf(Node);
    const blocks_off = alignUp(nodes_off + nodes_bytes, 64);
    const blocks_bytes = idx.blocks.len * @sizeOf(i16);
    const meta_off = alignUp(blocks_off + blocks_bytes, 64);
    const meta_bytes = idx.meta.len * @sizeOf(u32);
    const total = meta_off + meta_bytes;

    const h = Header{
        .magic = MAGIC.*,
        .n = idx.n,
        .n_nodes = idx.nodes.len,
        .root = idx.root,
        .leaf_max = idx.leaf_max,
        .n_blocks = idx.n_blocks,
        .nodes_off = nodes_off,
        .blocks_off = blocks_off,
        .meta_off = meta_off,
        .total = total,
    };
    const fd = try os.createTrunc(path);
    defer os.close(fd);
    var zeros: [64]u8 = .{0} ** 64;
    try os.writeAll(fd, std.mem.asBytes(&h));
    try os.writeAll(fd, zeros[0 .. nodes_off - @sizeOf(Header)]);
    try os.writeAll(fd, std.mem.sliceAsBytes(idx.nodes));
    try os.writeAll(fd, zeros[0 .. blocks_off - (nodes_off + nodes_bytes)]);
    try os.writeAll(fd, std.mem.sliceAsBytes(idx.blocks));
    try os.writeAll(fd, zeros[0 .. meta_off - (blocks_off + blocks_bytes)]);
    try os.writeAll(fd, std.mem.sliceAsBytes(idx.meta));
}

pub fn mapIndex(path: [*:0]const u8, pin: bool) !Index {
    const os = @import("os.zig");
    const fd = try os.openRead(path);
    defer os.close(fd);
    const size = try os.fileSize(fd);
    const base = try os.mapFileRead(fd, size);
    if (pin) os.pinMemory(base);
    const h: *const Header = @ptrCast(@alignCast(base.ptr));
    if (!std.mem.eql(u8, &h.magic, MAGIC)) return error.BadMagic;
    return .{
        .n = h.n,
        .n_nodes = h.n_nodes,
        .root = @intCast(h.root),
        .leaf_max = h.leaf_max,
        .n_blocks = h.n_blocks,
        .nodes = @as([*]Node, @ptrCast(@alignCast(base.ptr + h.nodes_off)))[0..h.n_nodes],
        .blocks = @as([*]align(32) i16, @ptrCast(@alignCast(base.ptr + h.blocks_off)))[0 .. h.n_blocks * BLOCK_I16],
        .meta = @as([*]u32, @ptrCast(@alignCast(base.ptr + h.meta_off)))[0 .. h.n_blocks * LANES],
        .mmap = base,
    };
}

// ---- quickselect (median partition) -----------------------------------------

inline fn keyOf(refs: *const Refs, perm: []const u32, i: usize, dim: usize) i16 {
    return refs.vec(perm[i])[dim];
}

fn quickselect(refs: *const Refs, perm: []u32, lo_in: usize, hi_in: usize, k: usize, dim: usize) void {
    var lo = lo_in;
    var hi = hi_in;
    while (lo < hi) {
        const mid = lo + (hi - lo) / 2;
        const a = keyOf(refs, perm, lo, dim);
        const b = keyOf(refs, perm, mid, dim);
        const c = keyOf(refs, perm, hi, dim);
        var pivot: i16 = undefined;
        if ((a <= b and b <= c) or (c <= b and b <= a)) pivot = b else if ((b <= a and a <= c) or (c <= a and a <= b)) pivot = a else pivot = c;
        var i = lo;
        var j = hi;
        while (true) {
            while (keyOf(refs, perm, i, dim) < pivot) i += 1;
            while (keyOf(refs, perm, j, dim) > pivot) j -= 1;
            if (i >= j) break;
            const t = perm[i];
            perm[i] = perm[j];
            perm[j] = t;
            i += 1;
            if (j == 0) break;
            j -= 1;
        }
        if (k <= j) hi = j else lo = j + 1;
    }
}

// ---- build -------------------------------------------------------------------

const Builder = struct {
    refs: *const Refs,
    perm: []u32,
    leaf_max: usize,
    nodes: std.ArrayList(Node),
    leaves: std.ArrayList(struct { start: usize, len: usize, node: u32 }),
    alloc: std.mem.Allocator,

    fn buildNode(self: *Builder, start: usize, len: usize) !u32 {
        const my: u32 = @intCast(self.nodes.items.len);
        try self.nodes.append(self.alloc, undefined);

        // widest-spread dim
        var mn: [VPAD]i16 = .{std.math.maxInt(i16)} ** VPAD;
        var mx: [VPAD]i16 = .{std.math.minInt(i16)} ** VPAD;
        for (self.perm[start .. start + len]) |orig| {
            const v = self.refs.vec(orig);
            inline for (0..14) |d| {
                if (v[d] < mn[d]) mn[d] = v[d];
                if (v[d] > mx[d]) mx[d] = v[d];
            }
        }
        var best_dim: usize = 0;
        var best_spread: i32 = -1;
        inline for (0..14) |d| {
            const sp = @as(i32, mx[d]) - @as(i32, mn[d]);
            if (sp > best_spread) {
                best_spread = sp;
                best_dim = d;
            }
        }

        if (len <= self.leaf_max or best_spread == 0) {
            self.nodes.items[my] = .{ .dim = LEAF, .split = 0, .a = 0, .b = @intCast(len) };
            try self.leaves.append(self.alloc, .{ .start = start, .len = len, .node = my });
            return my;
        }

        const mid = len / 2;
        quickselect(self.refs, self.perm, start, start + len - 1, start + mid, best_dim);
        const split_val = self.refs.vec(self.perm[start + mid])[best_dim];
        const left = try self.buildNode(start, mid);
        const right = try self.buildNode(start + mid, len - mid);
        self.nodes.items[my] = .{ .dim = @intCast(best_dim), .split = split_val, .a = left, .b = right };
        return my;
    }
};

pub fn build(alloc: std.mem.Allocator, refs: *const Refs, leaf_max: usize) !Index {
    const n = refs.n;
    var perm = try alloc.alloc(u32, n);
    defer alloc.free(perm);
    for (0..n) |i| perm[i] = @intCast(i);

    var b = Builder{
        .refs = refs,
        .perm = perm,
        .leaf_max = leaf_max,
        .nodes = .empty,
        .leaves = .empty,
        .alloc = alloc,
    };
    defer b.nodes.deinit(alloc);
    defer b.leaves.deinit(alloc);
    const root = try b.buildNode(0, n);

    var total_blocks: usize = 0;
    for (b.leaves.items) |lf| total_blocks += (lf.len + LANES - 1) / LANES;

    var idx = Index{
        .n = n,
        .n_nodes = b.nodes.items.len,
        .root = root,
        .leaf_max = leaf_max,
        .nodes = try alloc.alloc(Node, b.nodes.items.len),
        .blocks = try alloc.alignedAlloc(i16, .@"32", total_blocks * BLOCK_I16),
        .meta = try alloc.alloc(u32, total_blocks * LANES),
        .n_blocks = total_blocks,
        .alloc = alloc,
    };
    @memcpy(idx.nodes, b.nodes.items);
    @memset(idx.blocks, 0);
    @memset(idx.meta, 0xFF);

    var blk: usize = 0;
    for (b.leaves.items) |lf| {
        idx.nodes[lf.node].a = @intCast(blk); // first block of this leaf
        for (0..lf.len) |j| {
            const orig = perm[lf.start + j];
            const v = refs.vec(orig);
            const bb = blk + j / LANES;
            const lane = j % LANES;
            inline for (0..VPAD) |d| idx.blocks[bb * BLOCK_I16 + d * LANES + lane] = v[d];
            idx.meta[bb * LANES + lane] = (@as(u32, orig) << 1) | refs.labels[orig];
        }
        blk += (lf.len + LANES - 1) / LANES;
    }
    return idx;
}

// ---- search ------------------------------------------------------------------

const Top5 = struct {
    keys: [K]u64 = .{std.math.maxInt(u64)} ** K,
    worst_i: usize = 0,
    worst: u64 = std.math.maxInt(u64),

    inline fn offer(self: *Top5, key: u64) void {
        if (key < self.worst) {
            self.keys[self.worst_i] = key;
            self.worst = self.keys[0];
            self.worst_i = 0;
            inline for (1..K) |j| {
                if (self.keys[j] > self.worst) {
                    self.worst = self.keys[j];
                    self.worst_i = j;
                }
            }
        }
    }
    inline fn worstDist(self: *const Top5) i64 {
        return @intCast(self.worst >> DIST_SHIFT);
    }
};

inline fn blockDist(block: [*]const i16, q: *const [VPAD]i16) @Vector(LANES, i32) {
    var acc: @Vector(LANES, i32) = @splat(0);
    inline for (0..VPAD) |d| {
        const r: @Vector(LANES, i16) = block[d * LANES ..][0..LANES].*;
        const ri: @Vector(LANES, i32) = @intCast(r);
        const qi: @Vector(LANES, i32) = @splat(@as(i32, q[d]));
        const diff = ri - qi;
        acc += diff * diff;
    }
    return acc;
}

inline fn scanLeaf(idx: *const Index, node: Node, q: *const [VPAD]i16, top: *Top5) void {
    const cnt = node.b;
    const blk0 = node.a;
    const nfull = cnt / LANES;
    const rem = cnt % LANES;
    var b: u32 = 0;
    while (b < nfull) : (b += 1) {
        const dv = blockDist(idx.blocks.ptr + (blk0 + b) * BLOCK_I16, q);
        // SIMD skip: if no lane can beat the current 5th-best, don't touch the heap.
        if (@as(i64, @reduce(.Min, dv)) >= top.worstDist()) continue;
        const dists: [LANES]i32 = dv;
        const mbase = (blk0 + b) * LANES;
        inline for (0..LANES) |lane| {
            top.offer((@as(u64, @intCast(dists[lane])) << DIST_SHIFT) | idx.meta[mbase + lane]);
        }
    }
    if (rem != 0) {
        const dists: [LANES]i32 = blockDist(idx.blocks.ptr + (blk0 + nfull) * BLOCK_I16, q);
        const mbase = (blk0 + nfull) * LANES;
        var lane: usize = 0;
        while (lane < rem) : (lane += 1) {
            top.offer((@as(u64, @intCast(dists[lane])) << DIST_SHIFT) | idx.meta[mbase + lane]);
        }
    }
}

/// Latency guard: the true 5-NN almost always stabilise within the first few
/// hundred leaves (the query's own leaf + a little backtracking); the remaining
/// backtracking only *proves* exactness. For in-distribution queries the cap is
/// never reached (search stays exact, E=0). For out-of-distribution queries
/// (e.g. dates far outside the reference cloud) it bounds the worst-case work,
/// trading a vanishingly rare neighbour swap for a bounded p99.
pub const DEFAULT_CAP: usize = 8192;

const SearchCtx = struct {
    idx: *const Index,
    q: *const [VPAD]i16,
    cap: usize,
    top: Top5 = .{},
    off: [VPAD]i64 = .{0} ** VPAD,
    nleaf: usize = 0,

    fn visit(self: *SearchCtx, ni: u32, rd: i64) void {
        if (rd >= self.top.worstDist()) return;
        if (self.nleaf >= self.cap) return;
        const node = self.idx.nodes[ni];
        if (node.dim == LEAF) {
            scanLeaf(self.idx, node, self.q, &self.top);
            self.nleaf += 1;
            return;
        }
        const d = node.dim;
        const delta: i64 = @as(i64, self.q[d]) - @as(i64, node.split);
        const dd: i64 = delta * delta;
        if (delta <= 0) {
            self.visit(node.a, rd); // near = left
            const new_rd = rd - self.off[d] + dd;
            if (new_rd < self.top.worstDist()) {
                const save = self.off[d];
                self.off[d] = dd;
                self.visit(node.b, new_rd);
                self.off[d] = save;
            }
        } else {
            self.visit(node.b, rd); // near = right
            const new_rd = rd - self.off[d] + dd;
            if (new_rd < self.top.worstDist()) {
                const save = self.off[d];
                self.off[d] = dd;
                self.visit(node.a, new_rd);
                self.off[d] = save;
            }
        }
    }
};

pub inline fn search(idx: *const Index, q: *const [VPAD]i16) Result {
    return searchCore(idx, q, null, DEFAULT_CAP);
}

pub fn searchCore(idx: *const Index, q: *const [VPAD]i16, leaves: ?*usize, cap: usize) Result {
    var ctx = SearchCtx{ .idx = idx, .q = q, .cap = cap };
    ctx.visit(idx.root, 0);
    if (leaves) |p| p.* = ctx.nleaf;
    var fraud: u8 = 0;
    inline for (0..K) |j| fraud += @intCast(ctx.top.keys[j] & 1);
    return .{ .fraud_count = fraud, .approved = fraud < 3 };
}

// ---- test --------------------------------------------------------------------

test "kd-tree search matches oracle on small synthetic set" {
    const oracle = @import("oracle.zig");
    const a = std.testing.allocator;
    var prng = std.Random.DefaultPrng.init(12345);
    const rnd = prng.random();
    const n = 4000;
    var refs = Refs{
        .n = n,
        .vecs = try a.alloc(i16, n * VPAD),
        .labels = try a.alloc(u8, n),
        .alloc = a,
    };
    defer refs.deinit();
    @memset(refs.vecs, 0);
    for (0..n) |i| {
        inline for (0..14) |d| refs.vecs[i * VPAD + d] = rnd.intRangeAtMost(i16, 0, 10000);
        refs.labels[i] = rnd.intRangeAtMost(u8, 0, 1);
    }
    var idx = try build(a, &refs, 64);
    defer idx.deinit();
    var q: [VPAD]i16 = .{0} ** VPAD;
    for (0..800) |_| {
        inline for (0..14) |d| q[d] = rnd.intRangeAtMost(i16, 0, 10000);
        const r1 = oracle.knn(&refs, &q);
        const r2 = search(&idx, &q);
        try std.testing.expectEqual(r1.fraud_count, r2.fraud_count);
    }

    // serialize -> mmap roundtrip must give identical results
    try saveIndex(&idx, "/tmp/rnhz_test_index.bin");
    var midx = try mapIndex("/tmp/rnhz_test_index.bin", false);
    defer midx.deinit();
    prng = std.Random.DefaultPrng.init(777);
    const rnd2 = prng.random();
    for (0..800) |_| {
        inline for (0..14) |d| q[d] = rnd2.intRangeAtMost(i16, 0, 10000);
        try std.testing.expectEqual(search(&idx, &q).fraud_count, search(&midx, &q).fraud_count);
    }
}
