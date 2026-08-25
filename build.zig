const std = @import("std");
const sources = @import("sources.zig");
const build_zon = @import("build.zig.zon");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const fetch_exe = b.addExecutable(.{
        .name = "fetch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/fetch.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });
    const archive = b.cache_root.join(b.allocator, &.{ "onnxruntime", openvino_archive }) catch @panic("OOM");
    const fetch = b.addRunArtifact(fetch_exe);
    fetch.has_side_effects = true;
    fetch.addArgs(&.{
        "--url",     openvino_url,
        "--out",     archive,
        "--sha256",  openvino_sha256,
        "--extract", openvinoRoot(b),
        "--label",   "OpenVINO " ++ openvino_version ++ " (111 MiB)",
    });
    b.step("fetch-openvino", "Download the pinned Intel OpenVINO release").dependOn(&fetch.step);

    const openvino = b.option(
        bool,
        "openvino",
        "Build the Intel NPU and GPU OpenVINO execution provider",
    ) orelse false;
    if (openvino and !target.query.isNative()) {
        std.log.err("an OpenVINO build runs on the host's libstdc++ and cannot cross-compile", .{});
        std.process.exit(1);
    }

    const parts = Parts.init(b, target, optimize, openvino);
    b.installArtifact(parts.runtime());
    if (parts.openvino) {
        const openvino_root = openvinoRoot(b);
        const openvino_include = b.pathJoin(&.{ openvino_root, "runtime", "include" });
        const openvino_lib = b.pathJoin(&.{ openvino_root, "runtime", "lib", "intel64" });
        const shared_mod = parts.cxxModule(parts.target);
        shared_mod.addCSourceFiles(.{
            .root = parts.ort.path("onnxruntime"),
            .files = &sources.ort_provider_host_sources,
            .flags = parts.flags,
        });
        parts.linkCxx(shared_mod);
        const shared = b.addLibrary(.{
            .name = "onnxruntime_providers_shared",
            .linkage = .dynamic,
            .root_module = shared_mod,
        });
        shared.setVersionScript(parts.ort.path("onnxruntime/core/providers/shared/version_script.lds"));
        b.installArtifact(shared);

        const provider_mod = parts.cxxModule(parts.target);
        provider_mod.addIncludePath(parts.protos);
        provider_mod.addIncludePath(.{ .cwd_relative = openvino_include });
        const provider_flags = concatFlags(b, &.{
            parts.flags,
            &.{
                "-DUSE_OVEP_NPU_MEMORY=1",
                "-DFILE_NAME=\"libonnxruntime_providers_openvino.so\"",
                "-Wno-elaborated-enum-class",
                "-Wno-c++11-narrowing",
            },
        });
        const ort_root = parts.ort.path("onnxruntime");
        provider_mod.addCSourceFiles(.{ .root = ort_root, .files = &sources.ort_openvino_sources, .flags = provider_flags });
        provider_mod.addCSourceFiles(.{ .root = ort_root, .files = &sources.ort_provider_shared_sources, .flags = provider_flags });
        provider_mod.addCSourceFiles(.{ .root = parts.protos, .files = &sources.onnx_proto_sources, .flags = provider_flags });
        provider_mod.addCSourceFiles(.{ .root = parts.protobuf.path(""), .files = &sources.protobuf_lite_sources, .flags = provider_flags });
        provider_mod.addCSourceFiles(.{ .root = parts.abseil.path(""), .files = &sources.abseil_sources, .flags = provider_flags });
        provider_mod.addLibraryPath(.{ .cwd_relative = openvino_lib });
        provider_mod.linkSystemLibrary("openvino", .{});
        provider_mod.addRPath(.{ .cwd_relative = openvino_lib });
        parts.linkCxx(provider_mod);
        const provider = b.addLibrary(.{
            .name = "onnxruntime_providers_openvino",
            .linkage = .dynamic,
            .root_module = provider_mod,
        });
        provider.setVersionScript(parts.ort.path("onnxruntime/core/providers/openvino/version_script.lds"));
        provider.root_module.linkLibrary(shared);
        provider.root_module.addRPath(.{ .cwd_relative = "$ORIGIN" });
        provider.step.dependOn(&fetch.step);
        b.installArtifact(provider);

        const headers = b.dependency("opencl_headers", .{});
        const source = b.dependency("opencl_icd_loader", .{});
        const config = b.addWriteFiles();
        _ = config.add("icd_cmake_config.h", "#define HAVE_SECURE_GETENV\n");
        const opencl_mod = b.createModule(.{
            .target = target,
            .optimize = optimize,
            .link_libc = true,
        });
        opencl_mod.addIncludePath(headers.path("."));
        opencl_mod.addIncludePath(source.path("include"));
        opencl_mod.addIncludePath(source.path("loader"));
        opencl_mod.addIncludePath(config.getDirectory());
        opencl_mod.addCSourceFiles(.{
            .root = source.path("loader"),
            .files = &.{
                "icd.c",
                "icd_dispatch.c",
                "icd_dispatch_generated.c",
                "icd_trace.c",
                "linux/icd_linux.c",
                "linux/icd_linux_library.c",
                "linux/icd_linux_envvars.c",
            },
            .flags = &opencl_loader_flags,
        });
        const opencl = b.addLibrary(.{
            .name = "OpenCL",
            .linkage = .dynamic,
            .version = .{ .major = 1, .minor = 0, .patch = 0 },
            .root_module = opencl_mod,
        });
        opencl.setVersionScript(source.path("loader/linux/icd_exports.map"));
        b.installArtifact(opencl);
    }
}

pub fn library(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    return Parts.init(b, target, optimize, false).runtime();
}

pub fn linkStdCxx(b: *std.Build, mod: *std.Build.Module) void {
    Gnu.detect(b).link(mod);
}

const openvino_version = "2026.3.0";
const openvino_build = "22451.bd8d6542e3c";
const openvino_archive = "openvino_toolkit_ubuntu24_" ++ openvino_version ++ "." ++ openvino_build ++ "_x86_64.tgz";
const openvino_url = "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.3/linux/" ++ openvino_archive;
const openvino_sha256 = "0fa43c270bb6bc17bcf8c2c5fc5aa4595377292d54ef38084d8a0390daf4af98";

fn openvinoRoot(b: *std.Build) []const u8 {
    return b.cache_root.join(b.allocator, &.{ "onnxruntime", "openvino" }) catch @panic("OOM");
}

pub fn openvinoRuntimeLibraryPaths(b: *std.Build) []const []const u8 {
    const root = openvinoRoot(b);
    const paths = b.allocator.alloc([]const u8, 2) catch @panic("OOM");
    paths[0] = b.pathJoin(&.{ root, "runtime", "lib", "intel64" });
    paths[1] = b.pathJoin(&.{ root, "runtime", "3rdparty", "tbb", "lib" });
    return paths;
}

pub const OpenVinoDevice = enum { npu, gpu, cpu };

pub fn addOpenVinoRuntimeEnvironment(
    b: *std.Build,
    run: *std.Build.Step.Run,
    device: OpenVinoDevice,
    extra: []const []const u8,
    known: []const []const u8,
    opencl_driver_path: ?[]const u8,
) void {
    if (device == .gpu) {
        const driver = opencl_driver_path orelse driver: {
            const has_registry = registry: {
                var dir = std.Io.Dir.openDirAbsolute(b.graph.io, "/etc/OpenCL/vendors", .{
                    .iterate = true,
                }) catch break :registry false;
                defer dir.close(b.graph.io);
                var it = dir.iterate();
                while (it.next(b.graph.io) catch break :registry false) |entry| {
                    if (std.mem.endsWith(u8, entry.name, ".icd")) break :registry true;
                }
                break :registry false;
            };
            if (has_registry) break :driver null;
            for ([_][]const []const u8{ extra, &device_library_dirs }) |list| {
                for (list) |root| {
                    for ([_][]const u8{ b.pathJoin(&.{ root, opencl_driver_subdir }), root }) |dir| {
                        if (hasLibrary(b, dir, opencl_driver)) break :driver b.pathJoin(&.{ dir, opencl_driver });
                    }
                }
            }
            break :driver null;
        };
        if (driver) |path| run.setEnvironmentVariable("OCL_ICD_FILENAMES", path);
    }

    var dirs: std.ArrayList([]const u8) = .empty;
    defer dirs.deinit(b.allocator);
    if (run.getEnvMap().get("LD_LIBRARY_PATH")) |inherited| {
        if (inherited.len != 0) dirs.append(b.allocator, inherited) catch @panic("OOM");
    }
    dirs.appendSlice(b.allocator, extra) catch @panic("OOM");
    dirs.appendSlice(b.allocator, known) catch @panic("OOM");

    if (device == .npu) for ([2][]const u8{ "libze_loader.so.1", "libze_intel_npu.so.1" }) |library_name| {
        for (extra) |dir| {
            if (hasLibrary(b, dir, library_name)) break;
        } else for (device_library_dirs) |dir| {
            if (!hasLibrary(b, dir, library_name)) continue;
            for (dirs.items) |seen| {
                if (std.mem.eql(u8, seen, dir)) break;
            } else dirs.append(b.allocator, dir) catch @panic("OOM");
            break;
        }
    };

    if (dirs.items.len != 0) run.setEnvironmentVariable(
        "LD_LIBRARY_PATH",
        std.mem.join(b.allocator, ":", dirs.items) catch @panic("OOM"),
    );
}

const device_library_dirs = [_][]const u8{
    "/run/opengl-driver/lib",
    "/run/current-system/sw/lib",
    "/usr/lib/x86_64-linux-gnu",
    "/usr/lib64",
    "/usr/lib",
};

const opencl_driver = "libigdrcl.so";
const opencl_driver_subdir = "intel-opencl";

fn hasLibrary(b: *std.Build, dir: []const u8, library_name: []const u8) bool {
    const path = b.pathJoin(&.{ dir, library_name });
    std.Io.Dir.accessAbsolute(b.graph.io, path, .{}) catch return false;
    return true;
}

const Gnu = struct {
    include: []const []const u8,
    libraries: [2][]const u8,

    fn link(self: Gnu, mod: *std.Build.Module) void {
        for (self.libraries) |lib| {
            mod.addObjectFile(.{ .cwd_relative = lib });
            mod.addRPath(.{ .cwd_relative = std.fs.path.dirname(lib).? });
        }
    }

    fn detect(b: *std.Build) Gnu {
        const cxx = b.graph.environ_map.get("CXX") orelse "c++";

        const verbose = run(b, b.fmt("{s} -E -x c++ /dev/null -v 2>&1", .{cxx}));
        var include: std.ArrayList([]const u8) = .empty;
        var lines = std.mem.splitScalar(u8, verbose, '\n');
        var listing = false;
        while (lines.next()) |line| {
            const dir = std.mem.trim(u8, line, " \t\r");
            if (std.mem.startsWith(u8, dir, "#include <...>")) listing = true;
            if (std.mem.startsWith(u8, dir, "End of search list")) break;
            if (listing and std.mem.indexOf(u8, dir, "/c++/") != null) {
                include.append(b.allocator, dir) catch @panic("OOM");
            }
        }
        if (include.items.len == 0) {
            std.log.err("{s} reported no libstdc++ include directories", .{cxx});
            std.process.exit(1);
        }

        var libraries: [2][]const u8 = undefined;
        for ([_][]const u8{ "libstdc++.so", "libgcc_s.so.1" }, &libraries) |file, *out| {
            const path = std.mem.trim(u8, run(b, b.fmt("{s} -print-file-name={s}", .{ cxx, file })), " \t\r\n");
            if (std.mem.eql(u8, path, file)) {
                std.log.err("{s} could not locate {s}", .{ cxx, file });
                std.process.exit(1);
            }
            out.* = path;
        }
        return .{ .include = include.items, .libraries = libraries };
    }

    fn run(b: *std.Build, command: []const u8) []const u8 {
        var code: u8 = undefined;
        return b.runAllowFail(&.{ "sh", "-c", command }, &code, .ignore) catch |err| {
            std.log.err("running `{s}`: {s}", .{ command, @errorName(err) });
            std.process.exit(1);
        };
    }
};

const Parts = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    openvino: bool,
    gnu: ?Gnu,
    ort: *std.Build.Dependency,
    protobuf: *std.Build.Dependency,
    onnx: *std.Build.Dependency,
    abseil: *std.Build.Dependency,
    re2: *std.Build.Dependency,
    cpuinfo: *std.Build.Dependency,
    protos: std.Build.LazyPath,
    includes: []const std.Build.LazyPath,
    flags: []const []const u8,

    fn init(
        b: *std.Build,
        target: std.Build.ResolvedTarget,
        optimize: std.builtin.OptimizeMode,
        openvino: bool,
    ) Parts {
        const ort = b.dependency("ort_src", .{});
        const protobuf = b.dependency("protobuf", .{});
        const onnx = b.dependency("onnx", .{});
        const abseil = b.dependency("abseil", .{});
        const re2 = b.dependency("re2", .{});
        const cpuinfo = b.dependency("cpuinfo", .{});

        const config = b.addWriteFiles();
        _ = config.add("onnxruntime_config.h", b.fmt(ort_config_header, .{build_zon.version}));

        const protoc_mod = b.createModule(.{
            .target = b.graph.host,
            .optimize = .ReleaseSmall,
            .link_libc = true,
            .link_libcpp = true,
        });
        protoc_mod.addIncludePath(protobuf.path("src"));
        const protoc_flags = [_][]const u8{
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
            protoc_mod.addCSourceFiles(.{ .root = protobuf.path(""), .files = list, .flags = &protoc_flags });
        }
        const protoc = b.addExecutable(.{ .name = "protoc", .root_module = protoc_mod });
        const generate = b.addRunArtifact(protoc);
        for ([_][]const u8{ "onnx-ml.proto", "onnx-operators-ml.proto", "onnx-data.proto" }) |proto| {
            generate.addFileArg(onnx.path(b.fmt("onnx/{s}", .{proto})));
        }
        generate.addArg("-I");
        generate.addDirectoryArg(onnx.path(""));
        generate.addArg("--cpp_out");
        const protos = generate.addOutputDirectoryArg("onnx-proto");

        const includes = b.allocator.dupe(std.Build.LazyPath, &.{
            config.getDirectory(),
            ort.path("include/onnxruntime"),
            ort.path("include/onnxruntime/core/session"),
            ort.path("onnxruntime"),
            ort.path("onnxruntime/core/mlas/inc"),
            ort.path("onnxruntime/core/mlas/lib"),
            ort.path("model_package/include"),
            onnx.path(""),
            abseil.path(""),
            re2.path(""),
            protobuf.path("src"),
            cpuinfo.path("include"),
            cpuinfo.path("src"),
            b.dependency("ort_eigen", .{}).path(""),
            b.dependency("flatbuffers", .{}).path("include"),
            b.dependency("date", .{}).path("include"),
            b.dependency("gsl", .{}).path("include"),
            b.dependency("mp11", .{}).path("include"),
            b.dependency("safeint", .{}).path(""),
            b.dependency("json", .{}).path("single_include"),
        }) catch @panic("OOM");

        const cxx_base = if (openvino) &ort_openvino_flags else &ort_flags;
        const cxx_flags = if (target.result.os.tag.isDarwin())
            concatFlags(b, &.{ cxx_base, &.{"-gline-tables-only"} })
        else
            cxx_base;

        return .{
            .b = b,
            .target = target,
            .optimize = optimize,
            .openvino = openvino,
            .gnu = if (openvino) Gnu.detect(b) else null,
            .ort = ort,
            .protobuf = protobuf,
            .onnx = onnx,
            .abseil = abseil,
            .re2 = re2,
            .cpuinfo = cpuinfo,
            .protos = protos,
            .includes = includes,
            .flags = cxx_flags,
        };
    }

    fn cxxModule(self: Parts, target: std.Build.ResolvedTarget) *std.Build.Module {
        const mod = self.b.createModule(.{
            .target = target,
            .optimize = self.optimize,
            .link_libc = true,
            .link_libcpp = self.gnu == null,
        });
        for (self.includes) |include| mod.addIncludePath(include);
        if (self.gnu) |gnu| {
            for (gnu.include) |dir| mod.addIncludePath(.{ .cwd_relative = dir });
        }
        return mod;
    }

    fn linkCxx(self: Parts, mod: *std.Build.Module) void {
        if (self.gnu) |gnu| gnu.link(mod);
    }

    fn runtime(self: Parts) *std.Build.Step.Compile {
        const b = self.b;
        const ort_root = self.ort.path("onnxruntime");
        const target_sources = sources.forTarget(self.target.result);

        const lib_mod = self.cxxModule(self.target);
        lib_mod.addIncludePath(self.protos);
        if (self.target.result.os.tag.isDarwin()) {
            lib_mod.linkFramework("Foundation", .{});
            lib_mod.linkFramework("CoreFoundation", .{});
        }

        lib_mod.addCSourceFiles(.{ .root = self.protos, .files = &sources.onnx_proto_sources, .flags = self.flags });
        lib_mod.addCSourceFiles(.{
            .root = self.onnx.path(""),
            .files = &sources.onnx_sources,
            .flags = concatFlags(b, &.{ self.flags, &.{"-D__ONNX_DISABLE_STATIC_REGISTRATION"} }),
        });
        lib_mod.addCSourceFiles(.{ .root = self.abseil.path(""), .files = &sources.abseil_sources, .flags = self.flags });
        lib_mod.addCSourceFiles(.{ .root = self.re2.path(""), .files = &sources.re2_sources, .flags = self.flags });
        lib_mod.addCSourceFiles(.{ .root = self.protobuf.path(""), .files = &sources.protobuf_lite_sources, .flags = self.flags });
        lib_mod.addCSourceFiles(.{ .root = self.cpuinfo.path(""), .files = target_sources.cpuinfo, .flags = &ort_c_flags });
        lib_mod.addCSourceFiles(.{
            .root = self.ort.path("model_package"),
            .files = &sources.model_package_sources,
            .flags = self.flags,
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
            target_sources.mlas,
            target_sources.device_discovery,
        }) |list| {
            lib_mod.addCSourceFiles(.{ .root = ort_root, .files = list, .flags = self.flags });
        }
        for (target_sources.file_flags) |override| {
            lib_mod.addCSourceFiles(.{
                .root = ort_root,
                .files = &.{override.file},
                .flags = concatFlags(b, &.{ self.flags, override.flags }),
            });
        }

        const lib = b.addLibrary(.{
            .name = "onnxruntime",
            .linkage = .static,
            .root_module = lib_mod,
        });

        for (target_sources.mlas_groups, 0..) |group, index| {
            var query = self.target.query;
            query.cpu_features_add.addFeatureSet(group.features);

            const group_mod = self.cxxModule(b.resolveTargetQuery(query));
            group_mod.addCSourceFiles(.{
                .root = ort_root,
                .files = group.files,
                .flags = concatFlags(b, &.{ self.flags, &mlas_group_flags, group.flags }),
            });
            lib.root_module.linkLibrary(b.addLibrary(.{
                .name = b.fmt("onnxruntime-mlas-{d}", .{index}),
                .linkage = .static,
                .root_module = group_mod,
            }));
        }

        lib.installHeadersDirectory(self.ort.path("include/onnxruntime/core/session"), "", .{});
        return lib;
    }
};

const opencl_loader_flags = [_][]const u8{
    "-std=gnu99",
    "-DCL_TARGET_OPENCL_VERSION=310",
    "-DCL_NO_NON_ICD_DISPATCH_EXTENSION_PROTOTYPES",
    "-DOPENCL_ICD_LOADER_VERSION_MAJOR=3",
    "-DOPENCL_ICD_LOADER_VERSION_MINOR=1",
    "-DOPENCL_ICD_LOADER_VERSION_REV=0",
    "-DCL_ENABLE_LAYERS",
    "-DCL_ENABLE_LOADER_MANAGED_DISPATCH",
    "-DCL_SHARED_BUILD",
};

fn concatFlags(b: *std.Build, parts: []const []const []const u8) []const []const u8 {
    return std.mem.concat(b.allocator, []const u8, parts) catch @panic("OOM");
}

const ort_base_flags = [_][]const u8{
    "-std=c++20",
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
    "-DPLATFORM_POSIX",
    "-fno-sanitize=undefined",
    "-DGOOGLE_PROTOBUF_NO_RTTI=1",
    "-w",
};

const ort_flags = ort_base_flags ++ [_][]const u8{
    "-fno-rtti",
    "-DORT_NO_RTTI",
};

const ort_openvino_flags = ort_base_flags ++ [_][]const u8{
    "-Wno-invalid-constexpr",
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
