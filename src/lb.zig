//! Load balancer: accept TCP connections on :9999 and hand each accepted client
//! socket to an API worker (round-robin) over a Unix SOCK_SEQPACKET channel using
//! SCM_RIGHTS. It never reads the HTTP payload — pure fd forwarding.
//!   lb <port> <api1.sock> <api2.sock> [...]

const std = @import("std");
const linux = std.os.linux;
const posix = std.posix;
const os = @import("os.zig");

const AF = linux.AF;
const SOCK = linux.SOCK;
const SO = linux.SO;
const SOL = linux.SOL;
const SCM = linux.SCM;
const TCP = linux.TCP;
const IPPROTO = linux.IPPROTO;

const MAX_APIS = 8;

const usize_align = @sizeOf(usize);
inline fn cmsgAlign(n: usize) usize {
    return (n + usize_align - 1) & ~@as(usize, usize_align - 1);
}
const cmsghdr_aligned = cmsgAlign(@sizeOf(linux.cmsghdr));
const ONE_FD_SPACE = cmsghdr_aligned + cmsgAlign(@sizeOf(i32));

fn sendFd(sock: i32, fd_to_send: i32) bool {
    var byte: [1]u8 = .{0x46};
    var iov = [1]posix.iovec_const{.{ .base = &byte, .len = 1 }};
    var ctrl: [ONE_FD_SPACE]u8 align(@alignOf(linux.cmsghdr)) = undefined;
    @memset(&ctrl, 0);
    const cmsg: *linux.cmsghdr = @ptrCast(@alignCast(&ctrl));
    cmsg.* = .{ .len = cmsghdr_aligned + @sizeOf(i32), .level = SOL.SOCKET, .type = SCM.RIGHTS };
    const data: [*]u8 = @as([*]u8, @ptrCast(&ctrl)) + cmsghdr_aligned;
    @memcpy(data[0..@sizeOf(i32)], std.mem.asBytes(&fd_to_send));
    const msg = linux.msghdr_const{
        .name = null,
        .namelen = 0,
        .iov = &iov,
        .iovlen = 1,
        .control = &ctrl,
        .controllen = cmsghdr_aligned + @sizeOf(i32),
        .flags = 0,
    };
    var spins: u32 = 0;
    while (true) {
        const rc = linux.sendmsg(sock, &msg, linux.MSG.NOSIGNAL);
        switch (posix.errno(rc)) {
            .SUCCESS => return true,
            .AGAIN, .INTR => {
                spins += 1;
                if (spins > 1_000_000) return false;
            },
            else => return false,
        }
    }
}

fn sleepMs(ms: u64) void {
    var ts = linux.timespec{ .sec = @intCast(ms / 1000), .nsec = @intCast((ms % 1000) * 1_000_000) };
    _ = linux.nanosleep(&ts, &ts);
}

fn connectApi(path: [:0]const u8) !i32 {
    var attempt: u32 = 0;
    while (attempt < 600) : (attempt += 1) {
        const fd: i32 = @intCast(try os.ok(linux.socket(AF.UNIX, SOCK.SEQPACKET | SOCK.CLOEXEC, 0)));
        var un = linux.sockaddr.un{ .path = undefined };
        @memset(&un.path, 0);
        @memcpy(un.path[0..path.len], path);
        const un_len: linux.socklen_t = @intCast(@offsetOf(linux.sockaddr.un, "path") + path.len + 1);
        if (posix.errno(linux.connect(fd, @ptrCast(&un), un_len)) == .SUCCESS) return fd;
        _ = linux.close(fd);
        sleepMs(100);
    }
    return error.ConnectFailed;
}

fn tcpListener(port: u16) !i32 {
    const fd: i32 = @intCast(try os.ok(linux.socket(AF.INET, SOCK.STREAM | SOCK.CLOEXEC, IPPROTO.TCP)));
    const one: c_int = 1;
    const ob = std.mem.asBytes(&one);
    _ = linux.setsockopt(fd, SOL.SOCKET, SO.REUSEADDR, ob, ob.len);
    var addr = linux.sockaddr.in{ .port = std.mem.nativeToBig(u16, port), .addr = 0 };
    if (posix.errno(linux.bind(fd, @ptrCast(&addr), @sizeOf(linux.sockaddr.in))) != .SUCCESS) return error.Bind;
    if (posix.errno(linux.listen(fd, 1024)) != .SUCCESS) return error.Listen;
    return fd;
}

pub fn main(init: std.process.Init.Minimal) !void {
    var it = std.process.Args.Iterator.init(init.args);
    _ = it.next();
    const port_s = it.next() orelse return error.Args;
    const port = try std.fmt.parseInt(u16, port_s, 10);

    var api_fds: [MAX_APIS]i32 = undefined;
    var napis: usize = 0;
    while (it.next()) |path| {
        if (napis >= MAX_APIS) break;
        api_fds[napis] = try connectApi(path);
        napis += 1;
    }
    if (napis == 0) return error.NoApis;

    const lfd = try tcpListener(port);
    const one: c_int = 1;
    const ob = std.mem.asBytes(&one);

    var rr: usize = 0;
    while (true) {
        const arc = linux.accept4(lfd, null, null, SOCK.NONBLOCK | SOCK.CLOEXEC);
        switch (posix.errno(arc)) {
            .SUCCESS => {},
            .INTR, .AGAIN => continue,
            else => continue,
        }
        const client: i32 = @intCast(arc);
        _ = linux.setsockopt(client, IPPROTO.TCP, TCP.NODELAY, ob, ob.len);
        _ = sendFd(api_fds[rr], client);
        _ = linux.close(client);
        rr += 1;
        if (rr >= napis) rr = 0;
    }
}
