const std = @import("std");

pub fn build(b: *std.Build) void {
    // Deployment target: static musl, Haswell (AVX2, the test box's microarch).
    const target = b.resolveTargetQuery(.{
        .cpu_arch = .x86_64,
        .os_tag = .linux,
        .abi = .musl,
        .cpu_model = .{ .explicit = &std.Target.x86.cpu.haswell },
    });
    const optimize: std.builtin.OptimizeMode = .ReleaseFast;

    const exes = [_]struct { name: []const u8, src: []const u8 }{
        .{ .name = "lb", .src = "src/lb.zig" },
        .{ .name = "api", .src = "src/api.zig" },
        .{ .name = "indexer", .src = "src/indexer.zig" },
        .{ .name = "measure", .src = "src/measure.zig" },
    };

    for (exes) |e| {
        const mod = b.createModule(.{
            .root_source_file = b.path(e.src),
            .target = target,
            .optimize = optimize,
            .single_threaded = true,
            .strip = true,
        });
        const exe = b.addExecutable(.{ .name = e.name, .root_module = mod });
        b.installArtifact(exe);
    }

    // Unit tests (native).
    const test_step = b.step("test", "Run unit tests");
    for ([_][]const u8{ "src/vectorize.zig", "src/index.zig", "src/http.zig", "src/oracle.zig", "src/refs.zig", "src/os.zig" }) |src| {
        const tmod = b.createModule(.{ .root_source_file = b.path(src), .target = b.graph.host, .optimize = .Debug });
        const t = b.addTest(.{ .root_module = tmod });
        test_step.dependOn(&b.addRunArtifact(t).step);
    }
}
