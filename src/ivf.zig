//! IVF (inverted file) k-NN: k-means clusters the 3M int16 vectors into K tight,
//! contiguous cells. A query computes a lower bound to every cell's bounding box,
//! probes the nprobe nearest cells (contiguous SIMD scan), and keeps the 5 best.
//! Tight clusters concentrate a query's true neighbours in a few cells, so far
//! fewer vectors are examined than a kd-tree (~17k vs ~130k for out-of-distribution
//! queries) and the scan is cache-friendly — the key to sub-ms p99.

const std = @import("std");
const refs_mod = @import("refs.zig");
const Refs = refs_mod.Refs;
const idxmod = @import("index.zig");
const VPAD = @import("vectorize.zig").VPAD;
const os = @import("os.zig");

pub const K = idxmod.K; // 5 neighbours
pub const LANES = idxmod.LANES;
pub const DIST_SHIFT = idxmod.DIST_SHIFT;
const Top5 = idxmod.Top5;

// Pair-SoA block layout for the vpmaddwd distance kernel (jhon2c's trick): each
// block holds LANES (8) vectors over the 14 real dims, packed as 7 pairs. Pair p
// stores dims (2p, 2p+1) of the 8 vectors interleaved:
//   [v0_2p, v0_2p+1, v1_2p, v1_2p+1, ... v7_2p, v7_2p+1]   (16 i16 = one 256-bit reg)
// vpmaddwd(diff, diff) then yields, per vector j, (d_2p)^2 + (d_2p+1)^2 in one
// instruction (fused i16*i16 + horizontal pair-add) at ~1/2 the ops of the SoA-8
// widen+vmulld path, dropping the 2 zero pad dims entirely.
pub const PAIRS = 7; // 14 real dims / 2
pub const PBLOCK = PAIRS * 16; // 112 i16 per block

const QPairs = [PAIRS]@Vector(16, i16);

inline fn packQ(q: *const [VPAD]i16) QPairs {
    var qp: QPairs = undefined;
    inline for (0..PAIRS) |p| {
        var v: [16]i16 = undefined;
        const a = q[2 * p];
        const b = q[2 * p + 1];
        inline for (0..LANES) |lane| {
            v[lane * 2] = a;
            v[lane * 2 + 1] = b;
        }
        qp[p] = v;
    }
    return qp;
}

inline fn pmaddwd(d: @Vector(16, i16)) @Vector(8, i32) {
    return asm ("vpmaddwd %[d], %[d], %[o]"
        : [o] "=x" (-> @Vector(8, i32)),
        : [d] "x" (d),
    );
}

// Squared distance of 8 vectors (one block) to the query. Bit-identical to the
// SoA-8 blockDist over the 14 real dims; result lane j = vector j's distance.
inline fn dist8(block: [*]const i16, qp: *const QPairs) @Vector(8, i32) {
    var acc: @Vector(8, i32) = @splat(0);
    inline for (0..PAIRS) |p| {
        const bp: @Vector(16, i16) = block[p * 16 ..][0..16].*;
        acc += pmaddwd(bp -% qp[p]);
    }
    return acc;
}

pub const Result = idxmod.Result;

pub const Ivf = struct {
    n: usize,
    n_clusters: usize,
    blocks: []align(32) i16, // SoA-8 blocks, clusters contiguous
    meta: []u32, // block-ordered (orig_idx<<1)|label; pad = 0xFFFFFFFF
    cl_blk: []u32, // first block index per cluster
    cl_cnt: []u32, // real vector count per cluster
    bbox_min: []i16, // n_clusters*VPAD
    bbox_max: []i16, // n_clusters*VPAD
    n_blocks: usize = 0,
    alloc: ?std.mem.Allocator = null,
    owned: ?[]u8 = null,

    pub fn deinit(self: *Ivf) void {
        if (self.owned) |b| {
            if (self.alloc) |a| a.free(b);
        } else if (self.alloc) |a| {
            a.free(self.blocks);
            a.free(self.meta);
            a.free(self.cl_blk);
            a.free(self.cl_cnt);
            a.free(self.bbox_min);
            a.free(self.bbox_max);
        }
    }
};

// ---- k-means -----------------------------------------------------------------

inline fn sqdist16(a: *const [VPAD]i32, b: *const [VPAD]i16) i64 {
    var s: i64 = 0;
    inline for (0..VPAD) |d| {
        const df = a[d] - @as(i32, b[d]);
        s += @as(i64, df) * @as(i64, df);
    }
    return s;
}

/// Run k-means; returns assignment[n] (cluster id per point) and the centroids.
fn kmeans(alloc: std.mem.Allocator, refs: *const Refs, n_clusters: usize, iters: usize, assign: []u32) !void {
    const n = refs.n;
    // SoA centroids for fast assignment: cent[d*Kp + c]. Kp = n_clusters padded to 8.
    const Kp = (n_clusters + LANES - 1) / LANES * LANES;
    var cent = try alloc.alloc(i16, VPAD * Kp);
    defer alloc.free(cent);
    @memset(cent, 0);
    // init: stride-sampled points
    {
        const stride = n / n_clusters;
        var c: usize = 0;
        while (c < n_clusters) : (c += 1) {
            const v = refs.vec(c * stride);
            inline for (0..VPAD) |d| cent[d * Kp + c] = v[d];
        }
        // pad centroids -> far away so they never win
        var p = n_clusters;
        while (p < Kp) : (p += 1) inline for (0..VPAD) |d| {
            cent[d * Kp + p] = 30000;
        };
    }

    const sum = try alloc.alloc(i64, n_clusters * VPAD);
    defer alloc.free(sum);
    const cnt = try alloc.alloc(u32, n_clusters);
    defer alloc.free(cnt);

    var it: usize = 0;
    while (it < iters) : (it += 1) {
        // assign
        var i: usize = 0;
        while (i < n) : (i += 1) {
            const v = refs.vec(i);
            var best_d: i32 = std.math.maxInt(i32);
            var best_c: u32 = 0;
            var cg: usize = 0;
            while (cg < Kp) : (cg += LANES) {
                var acc: @Vector(LANES, i32) = @splat(0);
                inline for (0..VPAD) |d| {
                    const cc: @Vector(LANES, i16) = cent[d * Kp + cg ..][0..LANES].*;
                    const ci: @Vector(LANES, i32) = @intCast(cc);
                    const qd: @Vector(LANES, i32) = @splat(@as(i32, v[d]));
                    const df = ci - qd;
                    acc += df * df;
                }
                const arr: [LANES]i32 = acc;
                inline for (0..LANES) |l| {
                    if (arr[l] < best_d) {
                        best_d = arr[l];
                        best_c = @intCast(cg + l);
                    }
                }
            }
            assign[i] = if (best_c < n_clusters) best_c else 0;
        }
        if (it == iters - 1) break; // last assignment is final; skip update
        // update
        @memset(sum, 0);
        @memset(cnt, 0);
        i = 0;
        while (i < n) : (i += 1) {
            const c = assign[i];
            const v = refs.vec(i);
            inline for (0..VPAD) |d| sum[c * VPAD + d] += v[d];
            cnt[c] += 1;
        }
        var c: usize = 0;
        while (c < n_clusters) : (c += 1) {
            if (cnt[c] == 0) continue;
            inline for (0..VPAD) |d| {
                const m = @divTrunc(sum[c * VPAD + d], @as(i64, cnt[c]));
                cent[d * Kp + c] = @intCast(m);
            }
        }
    }
}

pub fn build(alloc: std.mem.Allocator, refs: *const Refs, n_clusters: usize, iters: usize) !Ivf {
    const n = refs.n;
    const assign = try alloc.alloc(u32, n);
    defer alloc.free(assign);
    try kmeans(alloc, refs, n_clusters, iters, assign);

    // counts + block layout
    const cnt = try alloc.alloc(u32, n_clusters);
    defer alloc.free(cnt);
    @memset(cnt, 0);
    for (assign) |c| cnt[c] += 1;

    var total_blocks: usize = 0;
    for (cnt) |c| total_blocks += (c + LANES - 1) / LANES;

    var ivf = Ivf{
        .n = n,
        .n_clusters = n_clusters,
        .blocks = try alloc.alignedAlloc(i16, .@"32", total_blocks * PBLOCK),
        .meta = try alloc.alloc(u32, total_blocks * LANES),
        .cl_blk = try alloc.alloc(u32, n_clusters),
        .cl_cnt = try alloc.alloc(u32, n_clusters),
        .bbox_min = try alloc.alloc(i16, n_clusters * VPAD),
        .bbox_max = try alloc.alloc(i16, n_clusters * VPAD),
        .n_blocks = total_blocks,
        .alloc = alloc,
    };
    @memset(ivf.blocks, 0);
    @memset(ivf.meta, 0xFF);

    // assign block ranges
    var blk: usize = 0;
    for (0..n_clusters) |c| {
        ivf.cl_blk[c] = @intCast(blk);
        ivf.cl_cnt[c] = cnt[c];
        blk += (cnt[c] + LANES - 1) / LANES;
        var d: usize = 0;
        while (d < VPAD) : (d += 1) {
            ivf.bbox_min[c * VPAD + d] = std.math.maxInt(i16);
            ivf.bbox_max[c * VPAD + d] = std.math.minInt(i16);
        }
    }
    // fill blocks + bbox; track per-cluster write cursor
    const wpos = try alloc.alloc(u32, n_clusters);
    defer alloc.free(wpos);
    @memset(wpos, 0);
    var i: usize = 0;
    while (i < n) : (i += 1) {
        const c = assign[i];
        const j = wpos[c];
        wpos[c] += 1;
        const b = ivf.cl_blk[c] + j / LANES;
        const lane = j % LANES;
        const v = refs.vec(i);
        // bbox over all VPAD dims (clusterLB is 16-wide; pad dims are 0).
        inline for (0..VPAD) |d| {
            if (v[d] < ivf.bbox_min[c * VPAD + d]) ivf.bbox_min[c * VPAD + d] = v[d];
            if (v[d] > ivf.bbox_max[c * VPAD + d]) ivf.bbox_max[c * VPAD + d] = v[d];
        }
        // block: pair-SoA over the 14 real dims (dims 14,15 are pad -> dropped).
        inline for (0..PAIRS) |p| {
            ivf.blocks[b * PBLOCK + p * 16 + lane * 2 + 0] = v[2 * p];
            ivf.blocks[b * PBLOCK + p * 16 + lane * 2 + 1] = v[2 * p + 1];
        }
        ivf.meta[b * LANES + lane] = (@as(u32, @intCast(i)) << 1) | refs.labels[i];
    }
    return ivf;
}

// ---- serialization -----------------------------------------------------------

const MAGIC = "RNHZIVF2"; // v2: pair-SoA blocks (PBLOCK=112) for the vpmaddwd kernel

pub const Header = extern struct {
    magic: [8]u8,
    n: u64,
    n_clusters: u64,
    n_blocks: u64,
    cl_blk_off: u64,
    cl_cnt_off: u64,
    bbox_min_off: u64,
    bbox_max_off: u64,
    blocks_off: u64,
    meta_off: u64,
    total: u64,
};

inline fn alignUp(x: usize, al: usize) usize {
    return (x + al - 1) & ~(al - 1);
}

pub fn save(ivf: *const Ivf, path: [*:0]const u8) !void {
    const nc = ivf.n_clusters;
    var off: usize = alignUp(@sizeOf(Header), 64);
    const cl_blk_off = off;
    off = alignUp(off + nc * 4, 64);
    const cl_cnt_off = off;
    off = alignUp(off + nc * 4, 64);
    const bbox_min_off = off;
    off = alignUp(off + nc * VPAD * 2, 64);
    const bbox_max_off = off;
    off = alignUp(off + nc * VPAD * 2, 64);
    const blocks_off = off;
    off = alignUp(off + ivf.blocks.len * 2, 64);
    const meta_off = off;
    const total = off + ivf.meta.len * 4;

    const h = Header{
        .magic = MAGIC.*,
        .n = ivf.n,
        .n_clusters = nc,
        .n_blocks = ivf.n_blocks,
        .cl_blk_off = cl_blk_off,
        .cl_cnt_off = cl_cnt_off,
        .bbox_min_off = bbox_min_off,
        .bbox_max_off = bbox_max_off,
        .blocks_off = blocks_off,
        .meta_off = meta_off,
        .total = total,
    };
    const fd = try os.createTrunc(path);
    defer os.close(fd);
    var z: [64]u8 = .{0} ** 64;
    var cur: usize = 0;
    const wr = struct {
        fn pad(f: i32, c: *usize, target: usize, zz: []u8) !void {
            while (c.* < target) {
                const n = @min(zz.len, target - c.*);
                try os.writeAll(f, zz[0..n]);
                c.* += n;
            }
        }
    };
    try os.writeAll(fd, std.mem.asBytes(&h));
    cur = @sizeOf(Header);
    try wr.pad(fd, &cur, cl_blk_off, &z);
    try os.writeAll(fd, std.mem.sliceAsBytes(ivf.cl_blk));
    cur += ivf.cl_blk.len * 4;
    try wr.pad(fd, &cur, cl_cnt_off, &z);
    try os.writeAll(fd, std.mem.sliceAsBytes(ivf.cl_cnt));
    cur += ivf.cl_cnt.len * 4;
    try wr.pad(fd, &cur, bbox_min_off, &z);
    try os.writeAll(fd, std.mem.sliceAsBytes(ivf.bbox_min));
    cur += ivf.bbox_min.len * 2;
    try wr.pad(fd, &cur, bbox_max_off, &z);
    try os.writeAll(fd, std.mem.sliceAsBytes(ivf.bbox_max));
    cur += ivf.bbox_max.len * 2;
    try wr.pad(fd, &cur, blocks_off, &z);
    try os.writeAll(fd, std.mem.sliceAsBytes(ivf.blocks));
    cur += ivf.blocks.len * 2;
    try wr.pad(fd, &cur, meta_off, &z);
    try os.writeAll(fd, std.mem.sliceAsBytes(ivf.meta));
}

pub fn map(alloc: std.mem.Allocator, path: [*:0]const u8, pin: bool) !Ivf {
    const buf = try os.readFileAlloc(alloc, path);
    errdefer alloc.free(buf);
    if (buf.len < @sizeOf(Header)) return error.Truncated;
    const h: *const Header = @ptrCast(@alignCast(buf.ptr));
    if (!std.mem.eql(u8, &h.magic, MAGIC)) return error.BadMagic;
    if (pin) _ = os.linux.mlockall(.{ .CURRENT = true, .FUTURE = true });
    const nc = h.n_clusters;
    return .{
        .n = h.n,
        .n_clusters = nc,
        .blocks = @as([*]align(32) i16, @ptrCast(@alignCast(buf.ptr + h.blocks_off)))[0 .. h.n_blocks * PBLOCK],
        .meta = @as([*]u32, @ptrCast(@alignCast(buf.ptr + h.meta_off)))[0 .. h.n_blocks * LANES],
        .cl_blk = @as([*]u32, @ptrCast(@alignCast(buf.ptr + h.cl_blk_off)))[0..nc],
        .cl_cnt = @as([*]u32, @ptrCast(@alignCast(buf.ptr + h.cl_cnt_off)))[0..nc],
        .bbox_min = @as([*]i16, @ptrCast(@alignCast(buf.ptr + h.bbox_min_off)))[0 .. nc * VPAD],
        .bbox_max = @as([*]i16, @ptrCast(@alignCast(buf.ptr + h.bbox_max_off)))[0 .. nc * VPAD],
        .n_blocks = h.n_blocks,
        .alloc = alloc,
        .owned = buf,
    };
}

// ---- search ------------------------------------------------------------------

inline fn clusterLB(q: *const [VPAD]i16, mn: []const i16, mx: []const i16) i64 {
    // Branchless 16-wide box lower bound: e = max(mn-q,0)+max(q-mx,0) per dim
    // (mn<=mx so at most one term is positive). Bit-identical to the scalar
    // version but ~25x faster — this LB runs over every cluster on every query,
    // so it dominated search time. Pad dims have mn=mx=q=0 -> contribute 0.
    const qv: @Vector(VPAD, i16) = q.*;
    const mnv: @Vector(VPAD, i16) = mn[0..VPAD].*;
    const mxv: @Vector(VPAD, i16) = mx[0..VPAD].*;
    const zero: @Vector(VPAD, i16) = @splat(0);
    const e: @Vector(VPAD, i16) = @max(mnv - qv, zero) + @max(qv - mxv, zero);
    const ei: @Vector(VPAD, i32) = e; // widen i16 -> i32 (e <= 20000)
    const sq: @Vector(VPAD, i64) = @intCast(ei * ei); // e^2 <= 4e8 per dim
    return @reduce(.Add, sq);
}

inline fn scanCluster(ivf: *const Ivf, c: usize, qp: *const QPairs, top: *Top5) void {
    const cnt = ivf.cl_cnt[c];
    const blk0 = ivf.cl_blk[c];
    const nfull = cnt / LANES;
    const rem = cnt % LANES;
    var b: u32 = 0;
    while (b < nfull) : (b += 1) {
        const dv = dist8(ivf.blocks.ptr + (blk0 + b) * PBLOCK, qp);
        if (@as(i64, @reduce(.Min, dv)) >= top.worstDist()) continue;
        const dists: [LANES]i32 = dv;
        const mbase = (blk0 + b) * LANES;
        inline for (0..LANES) |lane| {
            top.offer((@as(u64, @intCast(dists[lane])) << DIST_SHIFT) | ivf.meta[mbase + lane]);
        }
    }
    if (rem != 0) {
        const dists: [LANES]i32 = dist8(ivf.blocks.ptr + (blk0 + nfull) * PBLOCK, qp);
        const mbase = (blk0 + nfull) * LANES;
        var lane: usize = 0;
        while (lane < rem) : (lane += 1) {
            top.offer((@as(u64, @intCast(dists[lane])) << DIST_SHIFT) | ivf.meta[mbase + lane]);
        }
    }
}

const CL_BITS = 13; // cluster id fits (n_clusters <= 8192)

inline fn siftDown(h: []u64, n: usize, start: usize) void {
    var i = start;
    while (true) {
        const l = 2 * i + 1;
        const r = 2 * i + 2;
        var m = i;
        if (l < n and h[l] < h[m]) m = l;
        if (r < n and h[r] < h[m]) m = r;
        if (m == i) break;
        const t = h[i];
        h[i] = h[m];
        h[m] = t;
        i = m;
    }
}

/// Probe clusters in ascending box-lower-bound order. After `initial_probe` cells,
/// stop early when the decision is confident (0 or 5 frauds) or the next cell's
/// lower bound already exceeds the 5th-best distance (exact). Otherwise keep
/// probing up to `max_probe` — this adaptive expansion is what preserves E=0 on
/// the ambiguous out-of-distribution queries without paying for it on the common case.
/// `keys` scratch must be at least n_clusters long.
pub fn search(ivf: *const Ivf, q: *const [VPAD]i16, initial_probe: usize, max_probe: usize, keys: []u64, leaves: ?*usize) Result {
    const nc = ivf.n_clusters;
    for (0..nc) |c| {
        const lb = clusterLB(q, ivf.bbox_min[c * VPAD ..][0..VPAD], ivf.bbox_max[c * VPAD ..][0..VPAD]);
        keys[c] = (@as(u64, @intCast(lb)) << CL_BITS) | @as(u64, c);
    }
    // Min-heap selection instead of a full sort: probing usually stops after a
    // handful of cells, so sorting all nc cells is wasted work. Heapify once
    // (O(nc)) and pop the nearest cell on demand (O(log nc) each) — identical
    // ascending probe order, far less work per query.
    var hn = nc;
    {
        var i = nc / 2;
        while (i > 0) {
            i -= 1;
            siftDown(keys[0..nc], hn, i);
        }
    }

    const qp = packQ(q); // pack query into 7 pair-vectors once per search
    var top = Top5{};
    var probed: usize = 0;
    const mask: u64 = (1 << CL_BITS) - 1;
    while (probed < max_probe and hn > 0) {
        const kmin = keys[0];
        const lb: i64 = @intCast(kmin >> CL_BITS);
        if (lb >= top.worstDist()) break; // exact: no remaining cell can hold a closer point
        hn -= 1; // pop min
        keys[0] = keys[hn];
        siftDown(keys[0..nc], hn, 0);
        scanCluster(ivf, @intCast(kmin & mask), &qp, &top);
        probed += 1;
        if (probed >= initial_probe) {
            var fr: u8 = 0;
            inline for (0..K) |j| fr += @intCast(top.keys[j] & 1);
            if (fr == 0 or fr == K) break; // confident decision -> done
        }
    }
    if (leaves) |p| p.* = probed;
    var fraud: u8 = 0;
    inline for (0..K) |j| fraud += @intCast(top.keys[j] & 1);
    return .{ .fraud_count = fraud, .approved = fraud < 3 };
}
