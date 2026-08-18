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
const build_zon = @import("build.zig.zon");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    b.installArtifact(library(b, target, optimize));
}

/// A module compiled as C++ against Zig's libc++, seeded with `includes`.
/// Every module here wants the same libc/libc++ treatment; only the resolved
/// target and the include set differ.
fn cxxModule(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    includes: []const std.Build.LazyPath,
) *std.Build.Module {
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    for (includes) |include| mod.addIncludePath(include);
    return mod;
}

fn buildProtoc(b: *std.Build, protobuf: std.Build.LazyPath) *std.Build.Step.Compile {
    // Pinned rather than inherited from the caller: this protoc runs three
    // times on three small .proto files, so codegen quality buys nothing, and
    // -Os builds protobuf roughly twice as fast as any other mode. Pinning
    // also keeps the whole 167-file host build out of the consumer's
    // -Doptimize cache key, so it is paid once per host instead of per mode.
    const protoc_mod = cxxModule(b, b.graph.host, .ReleaseSmall, &.{protobuf.path(b, "src")});
    const flags = [_][]const u8{
        "-std=c++17",
        "-DGOOGLE_PROTOBUF_CMAKE_BUILD",
        "-DHAVE_ZLIB=0",
        "-w",
    };
    for ([_][]const []const u8{
        &sources.protobuf_lite_sources,
        &sources.protobuf_full_sources,
        &sources.protoc_sources,
        &.{"src/google/protobuf/compiler/main.cc"},
    }) |list| {
        protoc_mod.addCSourceFiles(.{ .root = protobuf, .files = list, .flags = &flags });
    }
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
    const ort_root = ort.path(b, "onnxruntime");

    const config = b.addWriteFiles();
    _ = config.add("onnxruntime_config.h", b.fmt(ort_config_header, .{build_zon.version}));

    // protoc runs during the build, so it is built for the host even when the
    // runtime itself is cross-compiled.
    const protos = generateOnnxProto(b, buildProtoc(b, protobuf), onnx);

    // Deliberately excludes `protos`. A generated include path is a step
    // dependency, so listing it here would park all of mlas behind protoc
    // instead of letting the two run at once; only the runtime proper needs
    // the generated headers, and it adds them for itself below.
    const includes = [_]std.Build.LazyPath{
        config.getDirectory(),
        ort.path(b, "include/onnxruntime"),
        ort.path(b, "include/onnxruntime/core/session"),
        ort_root,
        ort.path(b, "onnxruntime/core/mlas/inc"),
        ort.path(b, "onnxruntime/core/mlas/lib"),
        ort.path(b, "model_package/include"),
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
    };

    const lib_mod = cxxModule(b, target, optimize, &includes);
    lib_mod.addIncludePath(protos);

    lib_mod.addCSourceFiles(.{ .root = protos, .files = &sources.onnx_proto_sources, .flags = &ort_flags });
    lib_mod.addCSourceFiles(.{
        .root = onnx,
        .files = &sources.onnx_sources,
        .flags = concatFlags(b, &.{ &ort_flags, &.{"-D__ONNX_DISABLE_STATIC_REGISTRATION"} }),
    });
    lib_mod.addCSourceFiles(.{ .root = abseil, .files = &sources.abseil_sources, .flags = &ort_flags });
    lib_mod.addCSourceFiles(.{ .root = re2, .files = &sources.re2_sources, .flags = &ort_flags });
    lib_mod.addCSourceFiles(.{ .root = protobuf, .files = &sources.protobuf_lite_sources, .flags = &ort_flags });
    lib_mod.addCSourceFiles(.{ .root = cpuinfo, .files = &sources.cpuinfo_sources, .flags = &ort_c_flags });
    lib_mod.addCSourceFiles(.{
        .root = ort.path(b, "model_package"),
        .files = &sources.model_package_sources,
        .flags = &ort_flags,
    });

    for ([_][]const []const u8{
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
        lib_mod.addCSourceFiles(.{ .root = ort_root, .files = list, .flags = &ort_flags });
    }
    for (sources.ort_file_flags) |override| {
        lib_mod.addCSourceFiles(.{
            .root = ort_root,
            .files = &.{override.file},
            .flags = concatFlags(b, &.{ &ort_flags, override.flags }),
        });
    }

    const lib = b.addLibrary(.{
        .name = "onnxruntime",
        .linkage = .static,
        .root_module = lib_mod,
    });

    for (sources.ort_mlas_groups, 0..) |group, index| {
        var query = target.query;
        for (group.features) |feature| query.cpu_features_add.addFeature(@intFromEnum(feature));

        const group_mod = cxxModule(b, b.resolveTargetQuery(query), optimize, &includes);
        group_mod.addCSourceFiles(.{
            .root = ort_root,
            .files = group.files,
            .flags = concatFlags(b, &.{ &ort_flags, &mlas_group_flags, group.flags }),
        });
        lib.root_module.linkLibrary(b.addLibrary(.{
            .name = b.fmt("onnxruntime-mlas-{d}", .{index}),
            .linkage = .static,
            .root_module = group_mod,
        }));
    }

    // Travels with the artifact: a module that links this library gets the C
    // API headers on its include path without naming a path itself.
    lib.installHeadersDirectory(ort.path(b, "include/onnxruntime/core/session"), "", .{});
    return lib;
}

fn concatFlags(b: *std.Build, parts: []const []const []const u8) []const []const u8 {
    return std.mem.concat(b.allocator, []const u8, parts) catch @panic("OOM");
}

const ort_flags = [_][]const u8{
    "-std=c++20",
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
};

const ort_c_flags = [_][]const u8{
    "-std=c11",
    "-fno-sanitize=undefined",
    "-DCPUINFO_LOG_LEVEL=2",
    "-DCPUINFO_LOG_TO_STDIO=1",
    "-DCPUINFO_SUPPORTED",
    "-DCPUINFO_SUPPORTED_PLATFORM=1",
    "-D_GNU_SOURCE",
    "-w",
};

/// Applied to every mlas ISA group, on top of `ort_flags`.
const mlas_group_flags = [_][]const u8{
    "-fvisibility=hidden",
    "-fvisibility-inlines-hidden",
};

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
    \\#define ORT_VERSION "{s}"
    \\
;
