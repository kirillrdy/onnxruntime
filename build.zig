//! ONNX Runtime, built from source by the Zig build system.
//!
//! Everything the runtime needs -- protobuf and a host protoc, onnx, abseil,
//! re2, cpuinfo -- is compiled here from pinned source archives. Upstream's
//! CMake is not involved: the file lists live in sources.zig. The C++ runtime
//! comes from Zig's own libc++, so the result links against no system library
//! beyond libc, and cross-compiling needs nothing installed for the target.
//!
//!     const ort = b.dependency("onnxruntime", .{
//!         .target = target,
//!         .optimize = optimize,
//!     });
//!     exe.root_module.linkLibrary(ort.artifact("onnxruntime"));
//!
//! Linking the artifact brings its headers with it, so `@cInclude`ing
//! "onnxruntime_c_api.h" then works with no include path of your own.

const std = @import("std");
const sources = @import("sources.zig");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    b.installArtifact(library(b, target, optimize));
}

pub fn buildProtoc(b: *std.Build, protobuf: std.Build.LazyPath, optimize: std.builtin.OptimizeMode) *std.Build.Step.Compile {
    const protoc_mod = b.createModule(.{
        .target = b.graph.host,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    protoc_mod.addIncludePath(protobuf.path(b, "src"));
    const flags = [_][]const u8{
        "-std=c++17",
        "-DGOOGLE_PROTOBUF_CMAKE_BUILD",
        "-DHAVE_ZLIB=0",
        "-w",
    };
    inline for (.{ &sources.protobuf_lite_sources, &sources.protobuf_full_sources, &sources.protoc_sources }) |list| {
        protoc_mod.addCSourceFiles(.{ .root = protobuf, .files = list, .flags = &flags });
    }
    protoc_mod.addCSourceFiles(.{
        .root = protobuf,
        .files = &.{"src/google/protobuf/compiler/main.cc"},
        .flags = &flags,
    });
    return b.addExecutable(.{ .name = "protoc", .root_module = protoc_mod });
}

fn generateOnnxProto(b: *std.Build, protoc_exe: *std.Build.Step.Compile, onnx: std.Build.LazyPath) std.Build.LazyPath {
    const run = b.addRunArtifact(protoc_exe);
    for ([_][]const u8{ "onnx-ml.proto", "onnx-operators-ml.proto", "onnx-data.proto" }) |proto| {
        run.addFileArg(onnx.path(b, b.fmt("onnx/{s}", .{proto})));
    }
    run.addArg("-I");
    run.addDirectoryArg(onnx);
    run.addArg("--cpp_out");
    return run.addOutputDirectoryArg("onnx-proto");
}

/// The ONNX Runtime static library, headers included. Linking it is all a
/// dependent has to do.
pub fn library(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const ort = b.dependency("ort_src", .{}).path(".");
    const protobuf = b.dependency("protobuf", .{}).path(".");
    const onnx = b.dependency("onnx", .{}).path(".");
    const abseil = b.dependency("abseil", .{}).path(".");
    const re2 = b.dependency("re2", .{}).path(".");
    const cpuinfo = b.dependency("cpuinfo", .{}).path(".");

    const config = b.addWriteFiles();
    _ = config.add("onnxruntime_config.h", ort_config_header);
    const cpu_features = config.add("cpu_features2.c", ort_cpu_features2);

    // protoc runs during the build, so it is built for the host even when the
    // runtime itself is cross-compiled.
    const protos = generateOnnxProto(b, buildProtoc(b, protobuf, optimize), onnx);

    const includes = b.allocator.dupe(std.Build.LazyPath, &.{
        config.getDirectory(),
        protos,
        ort.path(b, "include/onnxruntime"),
        ort.path(b, "include/onnxruntime/core/session"),
        ort.path(b, "onnxruntime"),
        ort.path(b, "onnxruntime/core/mlas/inc"),
        ort.path(b, "onnxruntime/core/mlas/lib"),
        onnx,
        abseil,
        re2,
        protobuf.path(b, "src"),
        cpuinfo.path(b, "include"),
        cpuinfo.path(b, "src"),
        b.dependency("ort_eigen", .{}).path("."),
        b.dependency("flatbuffers", .{}).path("include"),
        b.dependency("date", .{}).path("include"),
        b.dependency("gsl", .{}).path("include"),
        b.dependency("mp11", .{}).path("include"),
        b.dependency("safeint", .{}).path("."),
        b.dependency("json", .{}).path("single_include"),
    }) catch @panic("OOM");

    const lib_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    for (includes) |include| lib_mod.addIncludePath(include);

    const flags = ortFlags(b);

    lib_mod.addCSourceFiles(.{ .root = protos, .files = &sources.onnx_proto_sources, .flags = flags });
    lib_mod.addCSourceFiles(.{
        .root = onnx,
        .files = &sources.onnx_sources,
        .flags = concatFlags(b, flags, &.{"-D__ONNX_DISABLE_STATIC_REGISTRATION"}),
    });
    lib_mod.addCSourceFiles(.{ .root = abseil, .files = &sources.abseil_sources, .flags = flags });
    lib_mod.addCSourceFiles(.{ .root = re2, .files = &sources.re2_sources, .flags = flags });
    lib_mod.addCSourceFiles(.{ .root = protobuf, .files = &sources.protobuf_lite_sources, .flags = flags });
    lib_mod.addCSourceFiles(.{ .root = cpuinfo, .files = &sources.cpuinfo_sources, .flags = ortCFlags(b) });
    lib_mod.addCSourceFile(.{ .file = cpu_features, .flags = &.{"-std=c11"} });

    const ort_root = ort.path(b, "onnxruntime");
    inline for (.{
        &sources.ort_common_sources,
        &sources.ort_graph_sources,
        &sources.ort_framework_sources,
        &sources.ort_optimizer_sources,
        &sources.ort_providers_sources,
        &sources.ort_session_sources,
        &sources.ort_util_sources,
        &sources.ort_lora_sources,
        &sources.ort_flatbuffers_sources,
        &sources.ort_mlas_sources,
    }) |list| {
        lib_mod.addCSourceFiles(.{ .root = ort_root, .files = list, .flags = flags });
    }
    for (sources.ort_file_flags) |override| {
        lib_mod.addCSourceFiles(.{
            .root = ort_root,
            .files = &.{override.file},
            .flags = concatFlags(b, flags, override.flags),
        });
    }

    var group_libs: std.ArrayList(*std.Build.Step.Compile) = .empty;
    for (sources.ort_mlas_groups, 0..) |group, index| {
        var query = target.query;
        for (group.features) |feature| query.cpu_features_add.addFeature(@intFromEnum(feature));

        const group_mod = b.createModule(.{
            .target = b.resolveTargetQuery(query),
            .optimize = optimize,
            .link_libc = true,
            .link_libcpp = true,
        });
        for (includes) |include| group_mod.addIncludePath(include);
        group_mod.addCSourceFiles(.{
            .root = ort_root,
            .files = group.files,
            .flags = concatFlags(b, flags, &.{ "-fvisibility=hidden", "-fvisibility-inlines-hidden" }),
        });
        group_libs.append(b.allocator, b.addLibrary(.{
            .name = b.fmt("onnxruntime-mlas-{d}", .{index}),
            .linkage = .static,
            .root_module = group_mod,
        })) catch @panic("OOM");
    }

    const lib = b.addLibrary(.{
        .name = "onnxruntime",
        .linkage = .static,
        .root_module = lib_mod,
    });
    for (group_libs.items) |group_lib| lib.root_module.linkLibrary(group_lib);

    // Travels with the artifact: a module that links this library gets the C
    // API headers on its include path without naming a path itself.
    lib.installHeadersDirectory(ort.path(b, "include/onnxruntime/core/session"), "", .{});
    return lib;
}

fn concatFlags(b: *std.Build, base: []const []const u8, extra: []const []const u8) []const []const u8 {
    const all = b.allocator.alloc([]const u8, base.len + extra.len) catch @panic("OOM");
    @memcpy(all[0..base.len], base);
    @memcpy(all[base.len..], extra);
    return all;
}

fn ortFlags(b: *std.Build) []const []const u8 {
    return b.allocator.dupe([]const u8, &.{
        "-std=c++17",
        "-fno-rtti",
        "-DCPUINFO_SUPPORTED",
        "-DCPUINFO_SUPPORTED_PLATFORM=1",
        "-DEIGEN_MPL2_ONLY",
        "-DEIGEN_USE_THREADS",
        "-DENABLE_CPU_FP16_TRAINING_OPS",
        "-D_GNU_SOURCE",
        "-DONLY_C_LOCALE=0",
        "-DONNX_ML=1",
        "-DONNX_NAMESPACE=onnx",
        "-D__ONNX_NO_DOC_STRINGS",
        "-DONNX_USE_LITE_PROTO=1",
        "-DORT_ENABLE_STREAM",
        "-DORT_NO_RTTI",
        "-DPLATFORM_POSIX",
        "-fno-sanitize=undefined",
        "-DGOOGLE_PROTOBUF_NO_RTTI=1",
        "-w",
    }) catch @panic("OOM");
}

fn ortCFlags(b: *std.Build) []const []const u8 {
    return b.allocator.dupe([]const u8, &.{
        "-std=c11",
        "-fno-sanitize=undefined",
        "-DCPUINFO_LOG_LEVEL=2",
        "-DCPUINFO_LOG_TO_STDIO=1",
        "-DCPUINFO_SUPPORTED",
        "-DCPUINFO_SUPPORTED_PLATFORM=1",
        "-D_GNU_SOURCE",
        "-w",
    }) catch @panic("OOM");
}

const ort_config_header =
    \\#pragma once
    \\
    \\#define HAS_ARRAY_BOUNDS
    \\#define HAS_BITWISE_INSTEAD_OF_LOGICAL
    \\#define HAS_CAST_FUNCTION_TYPE
    \\#define HAS_DEPRECATED_COPY
    \\#define HAS_DEPRECATED_DECLARATIONS
    \\#define HAS_DEPRECATED_LITERAL_OPERATOR
    \\#define HAS_DEPRECATED_THIS_CAPTURE
    \\#define HAS_FORMAT_TRUNCATION
    \\#define HAS_IGNORED_ATTRIBUTES
    \\#define HAS_MISSING_BRACES
    \\#define HAS_PARENTHESES
    \\#define HAS_REALLOCARRAY
    \\#define HAS_SHORTEN_64_TO_32
    \\#define HAS_TAUTOLOGICAL_POINTER_COMPARE
    \\#define HAS_UNUSED_BUT_SET_PARAMETER
    \\#define HAS_UNUSED_BUT_SET_VARIABLE
    \\#define HAS_UNUSED_VARIABLE
    \\#define ORT_BUILD_INFO "ORT Build Info: built by build.zig"
    \\#define ORT_VERSION "1.24.4"
    \\
;

const ort_cpu_features2 =
    \\/* cpuid_info.cc probes waitpkg through __builtin_cpu_supports, which reads a
    \\   table that gcc keeps in libgcc. Zig's compiler_rt carries __cpu_model but not
    \\   __cpu_features2, so define it here and leave it zeroed: onnxruntime then sees
    \\   no TPAUSE and spins the way it does on any cpu without the instruction. */
    \\unsigned int __cpu_features2[8];
    \\
;
