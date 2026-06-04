//! Fraud-score API worker. Receives already-accepted client sockets from the load
//! balancer over a Unix SOCK_SEQPACKET channel (SCM_RIGHTS), then serves HTTP/1.1
//! keep-alive directly on those sockets via a single-threaded epoll loop.
//!   api <unix_ctrl_path> <index.bin>

const std = @import("std");
const linux = std.os.linux;
const posix = std.posix;
const os = @import("os.zig");
const http = @import("http.zig");
const vec = @import("vectorize.zig");
const ivf = @import("ivf.zig");

const EPOLL = linux.EPOLL;
const AF = linux.AF;
const SOCK = linux.SOCK;
const SOL = linux.SOL;
const SCM = linux.SCM;

const MAX_FDS = 8192;
const BUF = 4096;

const Kind = enum(u8) { unused = 0, listen, ctrl, client };
const Conn = struct {
    kind: Kind = .unused,
    len: u32 = 0,
    buf: [BUF]u8 = undefined,
};

// IVF probe budget: probe INIT_PROBE clusters, then stop as soon as the decision
// is confident (0 or 5 frauds) or exact; expand up to MAX_PROBE only for the
// ambiguous out-of-distribution queries — keeping the common case sub-0.1ms.
const INIT_PROBE: usize = 24;
const MAX_PROBE: usize = 96;
const MAX_CLUSTERS: usize = 8192; // upper bound for the per-query LB-key scratch

var conns: [MAX_FDS]Conn = undefined;
var gidx: ivf.Ivf = undefined;
var gkeys: [MAX_CLUSTERS]u64 = undefined;
var epfd: i32 = undefined;

// epoll_wait with microsecond-precision timeout via epoll_pwait2 (kernel >= 5.11),
// falling back to millisecond epoll_wait on older kernels. timeout_us < 0 blocks.
inline fn epollWaitUs(events: []linux.epoll_event, timeout_us: i64) usize {
    if (timeout_us < 0) return linux.epoll_wait(epfd, events.ptr, @intCast(events.len), -1);
    if (timeout_us == 0) return linux.epoll_wait(epfd, events.ptr, @intCast(events.len), 0);
    var ts = linux.timespec{
        .sec = @intCast(@divTrunc(timeout_us, 1_000_000)),
        .nsec = @intCast(@mod(timeout_us, 1_000_000) * 1000),
    };
    const rc = linux.syscall6(.epoll_pwait2, @as(usize, @bitCast(@as(isize, epfd))), @intFromPtr(events.ptr), events.len, @intFromPtr(&ts), 0, 8);
    if (posix.errno(rc) == .NOSYS) {
        const ms: i32 = @intCast(@divTrunc(timeout_us + 999, 1000));
        return linux.epoll_wait(epfd, events.ptr, @intCast(events.len), ms);
    }
    return rc;
}

// Adaptive wait: poll, then busy-spin (PAUSE) for spin_ns catching back-to-back
// requests with no sleep, then block with a short finite timeout (idle_us) instead
// of forever — the periodic wakeup keeps the core warm (no deep C-state / freq
// drop), cutting the cold-wakeup latency that dominated the tail. (bmtec's loop.)
inline fn waitEvents(events: []linux.epoll_event, spin_ns: u64, idle_us: i64) usize {
    if (spin_ns == 0) return epollWaitUs(events, idle_us);
    var rc = epollWaitUs(events, 0);
    if (posix.errno(rc) == .SUCCESS and rc != 0) return rc;
    const start = os.nowNs();
    while (os.nowNs() - start < spin_ns) {
        rc = epollWaitUs(events, 0);
        if (posix.errno(rc) == .SUCCESS and rc != 0) return rc;
        std.atomic.spinLoopHint();
    }
    return epollWaitUs(events, idle_us);
}

inline fn epollAdd(fd: i32, events: u32) void {
    var ev = linux.epoll_event{ .events = events, .data = .{ .fd = fd } };
    _ = linux.epoll_ctl(epfd, EPOLL.CTL_ADD, fd, &ev);
}
inline fn epollDel(fd: i32) void {
    _ = linux.epoll_ctl(epfd, EPOLL.CTL_DEL, fd, null);
}

fn closeClient(fd: i32) void {
    epollDel(fd);
    _ = linux.close(fd);
    if (fd >= 0 and fd < MAX_FDS) conns[@intCast(fd)].kind = .unused;
}

// ---- SCM_RIGHTS receive -------------------------------------------------------

const usize_align = @sizeOf(usize);
inline fn cmsgAlign(n: usize) usize {
    return (n + usize_align - 1) & ~@as(usize, usize_align - 1);
}
const cmsghdr_aligned = cmsgAlign(@sizeOf(linux.cmsghdr));
const ONE_FD_SPACE = cmsghdr_aligned + cmsgAlign(@sizeOf(i32));

const RecvFd = union(enum) { fd: i32, again, closed };

fn recvOneFd(ctrl: i32) RecvFd {
    var byte: [1]u8 = undefined;
    var iov = [1]posix.iovec{.{ .base = &byte, .len = 1 }};
    var ctrlbuf: [ONE_FD_SPACE]u8 align(@alignOf(linux.cmsghdr)) = undefined;
    var msg = linux.msghdr{
        .name = null,
        .namelen = 0,
        .iov = &iov,
        .iovlen = 1,
        .control = &ctrlbuf,
        .controllen = ctrlbuf.len,
        .flags = 0,
    };
    const rc = linux.recvmsg(ctrl, &msg, 0);
    switch (posix.errno(rc)) {
        .SUCCESS => {},
        .AGAIN => return .again,
        else => return .closed,
    }
    if (rc == 0) return .closed;
    if (msg.controllen < @sizeOf(linux.cmsghdr)) return .again;
    const cmsg: *const linux.cmsghdr = @ptrCast(@alignCast(&ctrlbuf));
    if (cmsg.level != SOL.SOCKET or cmsg.type != SCM.RIGHTS) return .again;
    const data: [*]const u8 = @as([*]const u8, @ptrCast(&ctrlbuf)) + cmsghdr_aligned;
    var fd: i32 = undefined;
    @memcpy(std.mem.asBytes(&fd), data[0..@sizeOf(i32)]);
    return .{ .fd = fd };
}

// ---- HTTP handling ------------------------------------------------------------

fn writeAllSock(fd: i32, bytes: []const u8) bool {
    var off: usize = 0;
    var spins: u32 = 0;
    while (off < bytes.len) {
        const rc = linux.write(fd, bytes.ptr + off, bytes.len - off);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) return false;
                off += rc;
            },
            .AGAIN => {
                spins += 1;
                if (spins > 100000) return false;
            },
            .INTR => {},
            else => return false,
        }
    }
    return true;
}

/// Try to serve as many complete requests as are buffered in conns[fd].
/// Returns false if the connection should be closed.
fn process(fd: i32) bool {
    const c = &conns[@intCast(fd)];
    var consumed: usize = 0;
    while (true) {
        const data = c.buf[consumed..c.len];
        const hend_rel = http.headerEnd(data) orelse break; // need more bytes
        const headers = data[0..hend_rel];
        var total: usize = hend_rel;
        var resp: []const u8 = http.READY;
        if (data[0] == 'P') { // POST /fraud-score
            const cl = http.contentLength(headers) orelse 0;
            total = hend_rel + cl;
            if (data.len < total) break; // body incomplete
            const body = data[hend_rel..total];
            resp = if (vec.vectorize(body)) |q|
                http.RESP[ivf.search(&gidx, &q, INIT_PROBE, MAX_PROBE, gkeys[0..gidx.n_clusters], null).fraud_count]
            else
                http.SAFE;
        }
        if (!writeAllSock(fd, resp)) return false;
        consumed += total;
        if (consumed >= c.len) break;
    }
    if (consumed > 0) {
        const remaining = c.len - consumed;
        if (remaining > 0) std.mem.copyForwards(u8, c.buf[0..remaining], c.buf[consumed..c.len]);
        c.len = @intCast(remaining);
    }
    return true;
}

fn onClientReadable(fd: i32) void {
    const c = &conns[@intCast(fd)];
    while (true) {
        if (c.len >= BUF) {
            // request too large for our buffer; drop it safely
            c.len = 0;
        }
        const rc = linux.read(fd, c.buf[c.len..].ptr, BUF - c.len);
        switch (posix.errno(rc)) {
            .SUCCESS => {
                if (rc == 0) {
                    closeClient(fd);
                    return;
                }
                c.len += @intCast(rc);
                if (!process(fd)) {
                    closeClient(fd);
                    return;
                }
                // keep draining; level-triggered will refire but draining cuts syscalls
            },
            .AGAIN => return,
            .INTR => {},
            else => {
                closeClient(fd);
                return;
            },
        }
    }
}

fn onCtrlReadable(ctrl: i32) void {
    while (true) {
        switch (recvOneFd(ctrl)) {
            .fd => |cfd| {
                if (cfd < 0 or cfd >= MAX_FDS) {
                    _ = linux.close(cfd);
                    continue;
                }
                conns[@intCast(cfd)] = .{ .kind = .client, .len = 0 };
                const bp: c_int = 64; // SO_BUSY_POLL microseconds (best-effort)
                _ = linux.setsockopt(cfd, SOL.SOCKET, linux.SO.BUSY_POLL, std.mem.asBytes(&bp), @sizeOf(c_int));
                epollAdd(cfd, EPOLL.IN | EPOLL.RDHUP);
            },
            .again => return,
            .closed => return,
        }
    }
}

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    const ctrl_path = it.next() orelse return error.Args;
    const index_path = it.next() orelse return error.Args;
    // Optional epoll tuning: <spin_us> <idle_us>. Defaults (0, -1) = block forever
    // (original behavior). e.g. "30 80" = 30us spin then 80us finite-timeout block.
    const spin_us: u64 = if (it.next()) |s| (std.fmt.parseInt(u64, s, 10) catch 0) else 0;
    const idle_us: i64 = if (it.next()) |s| (std.fmt.parseInt(i64, s, 10) catch -1) else -1;
    const spin_ns: u64 = spin_us * 1000;

    gidx = try ivf.map(std.heap.page_allocator, index_path, true);

    // Unix SEQPACKET listener for the LB to connect and pass client fds.
    const lfd: i32 = @intCast(try os.ok(linux.socket(AF.UNIX, SOCK.SEQPACKET | SOCK.NONBLOCK | SOCK.CLOEXEC, 0)));
    var un = linux.sockaddr.un{ .path = undefined };
    @memset(&un.path, 0);
    const cp = ctrl_path;
    @memcpy(un.path[0..cp.len], cp);
    _ = linux.unlink(ctrl_path);
    const un_len: linux.socklen_t = @intCast(@offsetOf(linux.sockaddr.un, "path") + cp.len + 1);
    if (posix.errno(linux.bind(lfd, @ptrCast(&un), un_len)) != .SUCCESS) return error.Bind;
    if (posix.errno(linux.listen(lfd, 16)) != .SUCCESS) return error.Listen;

    epfd = @intCast(try os.ok(linux.epoll_create1(0)));
    // NAPI busy-poll on the epoll instance (kernel >= 6.9): epoll_wait polls the
    // device for up to busy_poll_usecs before sleeping, slashing wakeup latency
    // under CPU contention. Best-effort — ignored on older kernels. This is the
    // technique the #1 (asm) submission uses to reach sub-ms p99.
    const EpollParams = extern struct { busy_poll_usecs: u32, busy_poll_budget: u16, prefer_busy_poll: u8, pad: u8 };
    const EPIOCSPARAMS: u32 = 0x40086502; // _IOW('e', 0x02, sizeof(epoll_params)=8)
    var ep_params = EpollParams{ .busy_poll_usecs = 64, .busy_poll_budget = 8, .prefer_busy_poll = 1, .pad = 0 };
    _ = linux.ioctl(epfd, EPIOCSPARAMS, @intFromPtr(&ep_params));

    conns[@intCast(lfd)] = .{ .kind = .listen, .len = 0 };
    epollAdd(lfd, EPOLL.IN);

    var events: [256]linux.epoll_event = undefined;
    while (true) {
        const nrc = waitEvents(events[0..], spin_ns, idle_us);
        const n = switch (posix.errno(nrc)) {
            .SUCCESS => nrc,
            .INTR => continue,
            else => continue,
        };
        if (n == 0) continue; // finite idle timeout fired with no events; re-poll/spin
        for (events[0..n]) |e| {
            const fd = e.data.fd;
            if (fd < 0 or fd >= MAX_FDS) continue;
            switch (conns[@intCast(fd)].kind) {
                .listen => {
                    while (true) {
                        const arc = linux.accept4(fd, null, null, SOCK.NONBLOCK | SOCK.CLOEXEC);
                        if (posix.errno(arc) != .SUCCESS) break;
                        const ctrl: i32 = @intCast(arc);
                        if (ctrl >= MAX_FDS) {
                            _ = linux.close(ctrl);
                            continue;
                        }
                        conns[@intCast(ctrl)] = .{ .kind = .ctrl, .len = 0 };
                        epollAdd(ctrl, EPOLL.IN | EPOLL.RDHUP);
                    }
                },
                .ctrl => {
                    if (e.events & (EPOLL.HUP | EPOLL.ERR) != 0) {
                        epollDel(fd);
                        _ = linux.close(fd);
                        conns[@intCast(fd)].kind = .unused;
                        continue;
                    }
                    onCtrlReadable(fd);
                },
                .client => {
                    if (e.events & (EPOLL.HUP | EPOLL.ERR) != 0) {
                        closeClient(fd);
                        continue;
                    }
                    onClientReadable(fd);
                },
                .unused => {},
            }
        }
    }
}
