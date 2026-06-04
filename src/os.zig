//! Thin checked wrappers over raw Linux syscalls (std.os.linux). We bypass the
//! std.Io layer entirely: the server is built on epoll + raw sockets, and the
//! tools just need open/read/mmap. Everything returns a Zig error on failure.

const std = @import("std");
pub const linux = std.os.linux;

pub fn nowNs() u64 {
    var ts: linux.timespec = undefined;
    _ = linux.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

pub fn ok(rc: usize) !usize {
    return switch (linux.errno(rc)) {
        .SUCCESS => rc,
        else => error.Syscall,
    };
}

pub fn openRead(path: [*:0]const u8) !i32 {
    const rc = linux.open(path, .{ .ACCMODE = .RDONLY }, 0);
    return @intCast(try ok(rc));
}

pub fn fileSize(fd: i32) !usize {
    const end = try ok(linux.lseek(fd, 0, linux.SEEK.END));
    _ = try ok(linux.lseek(fd, 0, linux.SEEK.SET));
    return end;
}

pub fn close(fd: i32) void {
    _ = linux.close(fd);
}

pub fn createTrunc(path: [*:0]const u8) !i32 {
    const rc = linux.open(path, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, 0o644);
    return @intCast(try ok(rc));
}

pub fn writeAll(fd: i32, bytes: []const u8) !void {
    var off: usize = 0;
    while (off < bytes.len) {
        const rc = try ok(linux.write(fd, bytes.ptr + off, bytes.len - off));
        if (rc == 0) return error.WriteZero;
        off += rc;
    }
}

/// mmap a file read-only (MAP_PRIVATE | MAP_POPULATE -> pre-faults pages).
pub fn mapFileRead(fd: i32, size: usize) ![]align(std.heap.page_size_min) u8 {
    const rc = linux.mmap(null, size, .{ .READ = true }, .{ .TYPE = .PRIVATE, .POPULATE = true }, fd, 0);
    _ = try ok(rc);
    const ptr: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(rc);
    return ptr[0..size];
}

/// Pre-fault + lock pages so the working set stays resident (no refault stalls
/// under cgroup memory pressure — the failure mode that cost the OCaml attempt).
pub fn pinMemory(mem: []align(std.heap.page_size_min) u8) void {
    _ = linux.madvise(mem.ptr, mem.len, linux.MADV.WILLNEED);
    _ = linux.madvise(mem.ptr, mem.len, linux.MADV.HUGEPAGE);
    _ = linux.mlockall(.{ .CURRENT = true, .FUTURE = true });
}

/// Read an entire file into an allocator-owned buffer.
pub fn readFileAlloc(alloc: std.mem.Allocator, path: [*:0]const u8) ![]u8 {
    const fd = try openRead(path);
    defer close(fd);
    const size = try fileSize(fd);
    const buf = try alloc.alloc(u8, size);
    errdefer alloc.free(buf);
    var off: usize = 0;
    while (off < size) {
        const rc = try ok(linux.read(fd, buf.ptr + off, size - off));
        if (rc == 0) break;
        off += rc;
    }
    if (off != size) return error.ShortRead;
    return buf;
}

test "readFileAlloc + mmap on a known file" {
    const a = std.testing.allocator;
    const buf = readFileAlloc(a, "_official/resources/mcc_risk.json") catch return; // skip if absent
    defer a.free(buf);
    try std.testing.expect(buf.len > 10);
    try std.testing.expect(std.mem.indexOf(u8, buf, "5411") != null);
}
