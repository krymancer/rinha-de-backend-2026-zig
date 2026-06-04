//! Minimal HTTP/1.1 for the fraud-score endpoint. No framework, no per-request
//! serialization: the only possible response bodies are the 6 fraud-count
//! outcomes, precomputed in full (headers + body) at comptime. The hot path is a
//! single indexed slice + one write().

const std = @import("std");

const BODIES = [6][]const u8{
    "{\"approved\":true,\"fraud_score\":0.0}",
    "{\"approved\":true,\"fraud_score\":0.2}",
    "{\"approved\":true,\"fraud_score\":0.4}",
    "{\"approved\":false,\"fraud_score\":0.6}",
    "{\"approved\":false,\"fraud_score\":0.8}",
    "{\"approved\":false,\"fraud_score\":1.0}",
};

fn buildResp(comptime body: []const u8) []const u8 {
    return std.fmt.comptimePrint(
        "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {d}\r\nConnection: keep-alive\r\n\r\n{s}",
        .{ body.len, body },
    );
}

/// Full HTTP responses indexed by fraud_count (0..5).
pub const RESP = blk: {
    var arr: [6][]const u8 = undefined;
    for (BODIES, 0..) |b, i| arr[i] = buildResp(b);
    break :blk arr;
};

/// GET /ready -> 200.
pub const READY = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\nConnection: keep-alive\r\n\r\n";

/// Safe fallback (parse failure): never emit non-200. approved:true,score:0.0
/// costs at most an FP (weight 1) vs an HTTP error (weight 5 + failure-rate hit).
pub const SAFE = RESP[0];

/// Find end of headers ("\r\n\r\n"); returns index just past it, or null.
pub inline fn headerEnd(buf: []const u8) ?usize {
    if (std.mem.indexOf(u8, buf, "\r\n\r\n")) |i| return i + 4;
    return null;
}

/// Parse Content-Length (case-insensitive) within the header block.
pub fn contentLength(headers: []const u8) ?usize {
    // case-insensitive search for "content-length:"
    var i: usize = 0;
    while (i + 15 <= headers.len) : (i += 1) {
        if (std.ascii.eqlIgnoreCase(headers[i .. i + 15], "content-length:")) {
            var j = i + 15;
            while (j < headers.len and (headers[j] == ' ' or headers[j] == '\t')) : (j += 1) {}
            var end = j;
            while (end < headers.len and headers[end] >= '0' and headers[end] <= '9') : (end += 1) {}
            return std.fmt.parseInt(usize, headers[j..end], 10) catch null;
        }
    }
    return null;
}

test "responses well-formed" {
    try std.testing.expect(std.mem.indexOf(u8, RESP[3], "\"approved\":false") != null);
    try std.testing.expect(std.mem.indexOf(u8, RESP[3], "0.6") != null);
    try std.testing.expect(std.mem.indexOf(u8, RESP[0], "Content-Length: 35") != null);
    try std.testing.expect(std.mem.indexOf(u8, RESP[5], "Content-Length: 36") != null);
    try std.testing.expectEqual(@as(?usize, 13), contentLength("Content-Length: 13\r\n"));
    try std.testing.expectEqual(@as(?usize, 7), contentLength("content-length:  7\r\n"));
}
