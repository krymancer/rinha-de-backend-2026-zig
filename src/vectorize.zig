//! Payload (JSON) -> 14-dimension int16 query vector.
//!
//! Every coordinate the official data-generator emits is `round4`'d (rounded to
//! 4 decimals; see data-generator/main.c:758,797). Therefore multiplying by
//! 10000 and rounding to the nearest integer yields an EXACT int16 in
//! [-10000, 10000]. Squared-euclidean distance computed over these int16 values
//! is bit-identical (up to a 10000^2 scale factor) to the float64 brute force
//! the grader uses to label payloads. int16 here is lossless, not approximate.
//!
//! The sentinel -1 (dims 5 & 6 when `last_transaction` is null) becomes -10000,
//! the only value outside [0, 10000].

const std = @import("std");

pub const VDIM = 14;
pub const VPAD = 16; // padded to 16 for AVX2 (two trailing zero dims)

// normalization.json constants (fixed for the edition).
const MAX_AMOUNT: f64 = 10000.0;
const MAX_INSTALLMENTS: f64 = 12.0;
const AMOUNT_VS_AVG_RATIO: f64 = 10.0;
const MAX_MINUTES: f64 = 1440.0;
const MAX_KM: f64 = 1000.0;
const MAX_TX_24H: f64 = 20.0;
const MAX_MERCH_AVG: f64 = 10000.0;

inline fn clamp01(v: f64) f64 {
    return if (v < 0.0) 0.0 else if (v > 1.0) 1.0 else v;
}

/// round(v*10000) clamped to int16 sentinel range. Matches C round() (ties away
/// from zero), which is what @round does.
inline fn q(v: f64) i16 {
    const r = @round(v * 10000.0);
    const c = std.math.clamp(r, -10000.0, 10000.0);
    return @intFromFloat(c);
}

/// mcc_risk.json lookup; default 0.5 for unknown MCCs.
fn mccRisk(mcc: []const u8) f64 {
    const tbl = [_]struct { code: []const u8, risk: f64 }{
        .{ .code = "5411", .risk = 0.15 },
        .{ .code = "5812", .risk = 0.30 },
        .{ .code = "5912", .risk = 0.20 },
        .{ .code = "5944", .risk = 0.45 },
        .{ .code = "7801", .risk = 0.80 },
        .{ .code = "7802", .risk = 0.75 },
        .{ .code = "7995", .risk = 0.85 },
        .{ .code = "4511", .risk = 0.35 },
        .{ .code = "5311", .risk = 0.25 },
        .{ .code = "5999", .risk = 0.50 },
    };
    inline for (tbl) |e| {
        if (std.mem.eql(u8, mcc, e.code)) return e.risk;
    }
    return 0.5;
}

// ----- timestamp helpers (UTC) ------------------------------------------------

const Ts = struct { y: i64, mo: i64, d: i64, h: i64, mi: i64, s: i64 };

/// Parse "YYYY-MM-DDThh:mm:ssZ". Tolerant: uses parseInt on fixed offsets.
fn parseTs(s: []const u8) !Ts {
    if (s.len < 19) return error.BadTs;
    return .{
        .y = try std.fmt.parseInt(i64, s[0..4], 10),
        .mo = try std.fmt.parseInt(i64, s[5..7], 10),
        .d = try std.fmt.parseInt(i64, s[8..10], 10),
        .h = try std.fmt.parseInt(i64, s[11..13], 10),
        .mi = try std.fmt.parseInt(i64, s[14..16], 10),
        .s = try std.fmt.parseInt(i64, s[17..19], 10),
    };
}

/// days since 1970-01-01 (Howard Hinnant's algorithm == timegm semantics).
fn daysFromCivil(y_in: i64, m: i64, d: i64) i64 {
    const y = y_in - @as(i64, if (m <= 2) 1 else 0);
    const era = @divFloor(if (y >= 0) y else y - 399, 400);
    const yoe = y - era * 400; // [0, 399]
    const mp = @mod(m + 9, 12); // [0,11] with Mar=0
    const doy = @divTrunc(153 * mp + 2, 5) + d - 1; // [0,365]
    const doe = yoe * 365 + @divTrunc(yoe, 4) - @divTrunc(yoe, 100) + doy; // [0,146096]
    return era * 146097 + doe - 719468;
}

fn epochSecs(t: Ts) i64 {
    return daysFromCivil(t.y, t.mo, t.d) * 86400 + t.h * 3600 + t.mi * 60 + t.s;
}

/// day_of_week, Monday=0..Sunday=6 — replicates data-generator/main.c:366 exactly.
fn dayOfWeek(y_in: i64, m: i64, d: i64) i64 {
    const t = [_]i64{ 0, 3, 2, 5, 0, 3, 5, 1, 4, 6, 2, 4 };
    const y = y_in - @as(i64, if (m < 3) 1 else 0);
    const dow = @mod(y + @divTrunc(y, 4) - @divTrunc(y, 100) + @divTrunc(y, 400) + t[@intCast(m - 1)] + d, 7); // 0=Sun
    return @mod(dow + 6, 7); // 0=Mon
}

// ----- minimal JSON field extraction -----------------------------------------
// The payload schema is fixed (data-generator/main.c:request_to_json). We extract
// by key with brace/bracket/quote-aware value scanning — robust to field order
// and number formatting (we parse every number to f64).

/// Returns the value text for `"key":` within `buf` (first occurrence), or null.
/// For objects/arrays/strings the returned slice includes the delimiters.
fn field(buf: []const u8, comptime key: []const u8) ?[]const u8 {
    const needle = "\"" ++ key ++ "\":";
    const at = std.mem.indexOf(u8, buf, needle) orelse return null;
    var i = at + needle.len;
    // skip optional whitespace
    while (i < buf.len and (buf[i] == ' ' or buf[i] == '\t' or buf[i] == '\n' or buf[i] == '\r')) : (i += 1) {}
    if (i >= buf.len) return null;
    const start = i;
    const c = buf[i];
    if (c == '{' or c == '[') {
        const open = c;
        const close: u8 = if (c == '{') '}' else ']';
        var depth: i32 = 0;
        var in_str = false;
        while (i < buf.len) : (i += 1) {
            const ch = buf[i];
            if (in_str) {
                if (ch == '\\') {
                    i += 1;
                } else if (ch == '"') in_str = false;
            } else if (ch == '"') {
                in_str = true;
            } else if (ch == open) {
                depth += 1;
            } else if (ch == close) {
                depth -= 1;
                if (depth == 0) return buf[start .. i + 1];
            }
        }
        return null;
    } else if (c == '"') {
        i += 1;
        while (i < buf.len) : (i += 1) {
            if (buf[i] == '\\') {
                i += 1;
            } else if (buf[i] == '"') return buf[start .. i + 1];
        }
        return null;
    } else {
        while (i < buf.len and buf[i] != ',' and buf[i] != '}' and buf[i] != ']') : (i += 1) {}
        return buf[start..i];
    }
}

inline fn numField(buf: []const u8, comptime key: []const u8) ?f64 {
    const v = field(buf, key) orelse return null;
    return std.fmt.parseFloat(f64, v) catch null;
}

inline fn strInner(v: []const u8) []const u8 {
    if (v.len >= 2 and v[0] == '"' and v[v.len - 1] == '"') return v[1 .. v.len - 1];
    return v;
}

/// Vectorize a raw POST body into a 16-wide int16 vector (dims 14,15 = 0).
/// Returns null only if the payload is unparseable (caller emits a safe 200).
pub fn vectorize(buf: []const u8) ?[VPAD]i16 {
    const tx = field(buf, "transaction") orelse return null;
    const cust = field(buf, "customer") orelse return null;
    const merch = field(buf, "merchant") orelse return null;
    const term = field(buf, "terminal") orelse return null;

    const amount = numField(tx, "amount") orelse return null;
    const installments = numField(tx, "installments") orelse return null;
    const req_at = strInner(field(tx, "requested_at") orelse return null);
    const ts = parseTs(req_at) catch return null;

    const avg_amount = numField(cust, "avg_amount") orelse return null;
    const tx24 = numField(cust, "tx_count_24h") orelse return null;
    const known = field(cust, "known_merchants") orelse "[]";

    const merch_id = strInner(field(merch, "id") orelse return null);
    const mcc = strInner(field(merch, "mcc") orelse return null);
    const merch_avg = numField(merch, "avg_amount") orelse return null;

    const is_online = blk: {
        const v = field(term, "is_online") orelse return null;
        break :blk v.len > 0 and v[0] == 't';
    };
    const card_present = blk: {
        const v = field(term, "card_present") orelse return null;
        break :blk v.len > 0 and v[0] == 't';
    };
    const km_home = numField(term, "km_from_home") orelse return null;

    // last_transaction: null or {timestamp, km_from_current}
    var has_last = false;
    var minutes: f64 = -1.0;
    var last_km: f64 = -1.0;
    if (field(buf, "last_transaction")) |lt| {
        if (lt.len > 0 and lt[0] == '{') {
            const lt_ts = strInner(field(lt, "timestamp") orelse return null);
            const lk = numField(lt, "km_from_current") orelse return null;
            const tprev = parseTs(lt_ts) catch return null;
            has_last = true;
            minutes = @as(f64, @floatFromInt(epochSecs(ts) - epochSecs(tprev))) / 60.0;
            last_km = lk;
        }
    }

    // unknown_merchant: merchant.id NOT present in known_merchants.
    var idbuf: [40]u8 = undefined;
    const quoted = std.fmt.bufPrint(&idbuf, "\"{s}\"", .{merch_id}) catch return null;
    const known_flag: f64 = if (std.mem.indexOf(u8, known, quoted) != null) 1.0 else 0.0;
    const unknown_merchant: f64 = 1.0 - known_flag;

    const hour: f64 = @floatFromInt(ts.h);
    const dow: f64 = @floatFromInt(dayOfWeek(ts.y, ts.mo, ts.d));

    var out: [VPAD]i16 = .{0} ** VPAD;
    out[0] = q(clamp01(amount / MAX_AMOUNT));
    out[1] = q(clamp01(installments / MAX_INSTALLMENTS));
    out[2] = q(clamp01((amount / avg_amount) / AMOUNT_VS_AVG_RATIO));
    out[3] = q(hour / 23.0);
    out[4] = q(dow / 6.0);
    if (has_last) {
        out[5] = q(clamp01(minutes / MAX_MINUTES));
        out[6] = q(clamp01(last_km / MAX_KM));
    } else {
        out[5] = -10000;
        out[6] = -10000;
    }
    out[7] = q(clamp01(km_home / MAX_KM));
    out[8] = q(clamp01(tx24 / MAX_TX_24H));
    out[9] = if (is_online) 10000 else 0;
    out[10] = if (card_present) 10000 else 0;
    out[11] = q(unknown_merchant);
    out[12] = q(mccRisk(mcc));
    out[13] = q(clamp01(merch_avg / MAX_MERCH_AVG));
    return out;
}

// ----- tests ------------------------------------------------------------------

test "legit example (DETECTION_RULES.md)" {
    const p =
        \\{"id":"tx-1329056812","transaction":{"amount":41.12,"installments":2,"requested_at":"2026-03-11T18:45:53Z"},"customer":{"avg_amount":82.24,"tx_count_24h":3,"known_merchants":["MERC-003","MERC-016"]},"merchant":{"id":"MERC-016","mcc":"5411","avg_amount":60.25},"terminal":{"is_online":false,"card_present":true,"km_from_home":29.23},"last_transaction":null}
    ;
    const got = vectorize(p).?;
    const want = [_]i16{ 41, 1667, 500, 7826, 3333, -10000, -10000, 292, 1500, 0, 10000, 0, 1500, 60, 0, 0 };
    try std.testing.expectEqualSlices(i16, &want, &got);
}

test "fraud example (DETECTION_RULES.md)" {
    const p =
        \\{"id":"tx-3330991687","transaction":{"amount":9505.97,"installments":10,"requested_at":"2026-03-14T05:15:12Z"},"customer":{"avg_amount":81.28,"tx_count_24h":20,"known_merchants":["MERC-008","MERC-007","MERC-005"]},"merchant":{"id":"MERC-068","mcc":"7802","avg_amount":54.86},"terminal":{"is_online":false,"card_present":true,"km_from_home":952.27},"last_transaction":null}
    ;
    const got = vectorize(p).?;
    // [0.9506,0.8333,1.0,0.2174,0.8333,-1,-1,0.9523,1.0,0,1,1,0.75,0.0055]
    const want = [_]i16{ 9506, 8333, 10000, 2174, 8333, -10000, -10000, 9523, 10000, 0, 10000, 10000, 7500, 55, 0, 0 };
    try std.testing.expectEqualSlices(i16, &want, &got);
}

test "smoke example with last_transaction" {
    const p =
        \\{"id":"tx-smoke-001","transaction":{"amount":384.88,"installments":3,"requested_at":"2026-03-11T20:23:35Z"},"customer":{"avg_amount":769.76,"tx_count_24h":3,"known_merchants":["MERC-009","MERC-001","MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5912","avg_amount":298.95},"terminal":{"is_online":false,"card_present":true,"km_from_home":13.7090520965},"last_transaction":{"timestamp":"2026-03-11T14:58:35Z","km_from_current":18.8626479774}}
    ;
    const got = vectorize(p).?;
    const want = [_]i16{ 385, 2500, 500, 8696, 3333, 2257, 189, 137, 1500, 0, 10000, 0, 2000, 299, 0, 0 };
    try std.testing.expectEqualSlices(i16, &want, &got);
}

test "day_of_week matches generator" {
    // 2026-03-11 is Wednesday -> Mon-based 2
    try std.testing.expectEqual(@as(i64, 2), dayOfWeek(2026, 3, 11));
    // 2026-03-14 Saturday -> 5
    try std.testing.expectEqual(@as(i64, 5), dayOfWeek(2026, 3, 14));
}
