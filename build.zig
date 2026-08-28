const std = @import("std");
const sources = @import("sources.zig");
const openvino_sources = @import("openvino_sources.zig");
const openvino_onnx_sources = @import("openvino_onnx_sources.zig");
const build_zon = @import("build.zig.zon");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const openvino = b.option(
        bool,
        "openvino",
        "Build the Intel GPU OpenVINO execution provider from source",
    ) orelse false;
    if (openvino and (target.result.os.tag != .linux or target.result.cpu.arch != .x86_64)) {
        std.log.err("the native OpenVINO GPU build currently supports x86_64-linux only", .{});
        std.process.exit(1);
    }

    const native_openvino = if (openvino) buildOpenVino(b, target, optimize) else null;

    const parts = Parts.init(b, target, optimize, openvino);
    b.installArtifact(parts.runtime());
    if (parts.openvino) {
        const openvino_dep = b.dependency("openvino", .{});
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
        provider_mod.addIncludePath(openvino_dep.path(""));
        provider_mod.addIncludePath(b.path("openvino-compat"));
        provider_mod.addIncludePath(openvino_dep.path("src/core/include"));
        provider_mod.addIncludePath(openvino_dep.path("src/inference/include"));
        provider_mod.addIncludePath(openvino_dep.path("src/frontends/common/include"));
        provider_mod.addIncludePath(openvino_dep.path("src/frontends/onnx/frontend/include"));
        provider_mod.addIncludePath(openvino_dep.path("src/frontends/onnx/frontend/src"));
        provider_mod.addIncludePath(openvino_dep.path("src/frontends/onnx/onnx_common/include"));
        provider_mod.addIncludePath(openvino_dep.path("src/frontends/onnx/onnx_common/src"));
        for (openvino_include_dirs) |dir| provider_mod.addIncludePath(openvino_dep.path(dir));
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
        addOpenVinoGroup(b, provider_mod, openvino_dep, &openvino_onnx_sources.common, &openvino_onnx_frontend_flags);
        addOpenVinoGroup(b, provider_mod, openvino_dep, &openvino_onnx_sources.frontend, &openvino_onnx_frontend_flags);
        provider_mod.linkLibrary(native_openvino.?.runtime);
        provider_mod.linkLibrary(native_openvino.?.opencl);
        parts.linkCxx(provider_mod);
        const provider = b.addLibrary(.{
            .name = "onnxruntime_providers_openvino",
            .linkage = .dynamic,
            .root_module = provider_mod,
        });
        provider.setVersionScript(parts.ort.path("onnxruntime/core/providers/openvino/version_script.lds"));
        provider.root_module.linkLibrary(shared);
        provider.root_module.addRPath(.{ .cwd_relative = "$ORIGIN" });
        b.installArtifact(provider);
    }
}

pub fn library(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    return Parts.init(b, target, optimize, false).runtime();
}

pub fn linkStdCxx(_: *std.Build, _: *std.Build.Module) void {}

pub fn openvinoRuntimeLibraryPaths(b: *std.Build) []const []const u8 {
    const paths = b.allocator.alloc([]const u8, 1) catch @panic("OOM");
    paths[0] = b.getInstallPath(.lib, "");
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

const NativeOpenVino = struct {
    runtime: *std.Build.Step.Compile,
    opencl: *std.Build.Step.Compile,
};

fn buildOpenCl(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    const headers = b.dependency("opencl_headers", .{});
    const source = b.dependency("opencl_icd_loader", .{});
    const config = b.addWriteFiles();
    _ = config.add("icd_cmake_config.h", "#define HAVE_SECURE_GETENV\n");
    const mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
    });
    mod.addIncludePath(headers.path("."));
    mod.addIncludePath(source.path("include"));
    mod.addIncludePath(source.path("loader"));
    mod.addIncludePath(config.getDirectory());
    mod.addCSourceFiles(.{
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
        .root_module = mod,
    });
    opencl.setVersionScript(source.path("loader/linux/icd_exports.map"));
    b.installArtifact(opencl);
    return opencl;
}

fn addOpenVinoIncludes(
    b: *std.Build,
    mod: *std.Build.Module,
    ov: *std.Build.Dependency,
    generated_plugins: std.Build.LazyPath,
    kernel_db: KernelDb,
) void {
    mod.addIncludePath(ov.path(""));
    for (openvino_include_dirs) |dir| mod.addIncludePath(ov.path(dir));
    mod.addIncludePath(generated_plugins);
    mod.addIncludePath(b.path("openvino-compat"));
    mod.addIncludePath(kernel_db.kernel_selector);
    mod.addIncludePath(kernel_db.ocl_v2);
    mod.addIncludePath(b.dependency("json", .{}).path("single_include"));
    mod.addIncludePath(b.dependency("openvino_pugixml", .{}).path("src"));
    mod.addIncludePath(b.dependency("openvino_ittapi", .{}).path("include"));
    mod.addIncludePath(b.dependency("openvino_ittapi", .{}).path("src/ittnotify"));
    mod.addIncludePath(b.dependency("openvino_xbyak", .{}).path(""));
    mod.addIncludePath(b.dependency("opencl_headers", .{}).path(""));
    mod.addIncludePath(b.dependency("openvino_opencl_hpp", .{}).path("include"));
}

const KernelDb = struct {
    /// Holds ks_primitive_db.inc and ks_primitive_db_batch_headers.inc.
    kernel_selector: std.Build.LazyPath,
    /// Holds gpu_ocl_kernel_sources.inc and gpu_ocl_kernel_headers.inc.
    ocl_v2: std.Build.LazyPath,
};

/// Stringifies the GPU plugin's OpenCL kernels into the `.inc` databases its
/// sources `#include`. OpenVINO's CMake build generates these with two Python
/// scripts; `tools/cl_kernel_db.zig` is a port of them, so the databases come
/// out of the pinned sources rather than a snapshot that can drift from them.
fn buildKernelDb(b: *std.Build, ov: *std.Build.Dependency) KernelDb {
    const generator = b.addExecutable(.{
        .name = "cl_kernel_db",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/cl_kernel_db.zig"),
            .target = b.graph.host,
            .optimize = .ReleaseFast,
        }),
    });

    const kernels = ov.path("src/plugins/intel_gpu/src/kernel_selector/cl_kernels");

    const primitive_db = b.addRunArtifact(generator);
    primitive_db.addArg("primitive-db");
    primitive_db.addDirectoryArg(kernels);
    const kernel_selector = primitive_db.addOutputDirectoryArg("kernel_selector");

    const ocl_v2 = b.addRunArtifact(generator);
    ocl_v2.addArg("ocl-v2");
    ocl_v2.addDirectoryArg(ov.path("src/plugins/intel_gpu/src/graph/impls/ocl_v2"));
    ocl_v2.addDirectoryArg(kernels.path(b, "include"));

    return .{
        .kernel_selector = kernel_selector,
        .ocl_v2 = ocl_v2.addOutputDirectoryArg("ocl_v2"),
    };
}

fn addOpenVinoGroup(
    b: *std.Build,
    mod: *std.Build.Module,
    ov: *std.Build.Dependency,
    files: []const []const u8,
    extra_flags: []const []const u8,
) void {
    mod.addCSourceFiles(.{
        .root = ov.path(""),
        .files = files,
        .flags = concatFlags(b, &.{ &openvino_common_flags, extra_flags }),
    });
}

fn buildOpenVino(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) NativeOpenVino {
    const ov = b.dependency("openvino", .{});
    const opencl = buildOpenCl(b, target, optimize);
    const generated = b.addWriteFiles();
    const plugins_header = generated.add("ov_plugins.hpp", openvino_plugins_header);
    _ = generated.add("ov_frontends.hpp", openvino_frontends_header);

    const runtime_mod = b.createModule(.{
        .target = target,
        .optimize = optimize,
        .link_libc = true,
        .link_libcpp = true,
    });
    addOpenVinoIncludes(b, runtime_mod, ov, plugins_header.dirname(), buildKernelDb(b, ov));
    runtime_mod.linkSystemLibrary("dl", .{});

    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_core_obj, &openvino_core_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_core_obj_version, &openvino_core_version_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_frontend_common_obj, &openvino_frontend_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_transformations_obj, &openvino_api_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_lp_transformations_obj, &openvino_api_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_runtime_obj, &openvino_runtime_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_reference, &openvino_reference_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_shape_inference, &openvino_core_internal_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_itt, &.{});
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_shutdown, &.{});
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_util, &.{});
    runtime_mod.addCSourceFiles(.{
        .root = b.dependency("openvino_pugixml", .{}).path(""),
        .files = &.{"src/pugixml.cpp"},
        .flags = &openvino_common_flags,
    });
    runtime_mod.addCSourceFiles(.{
        .root = b.dependency("openvino_ittapi", .{}).path(""),
        .files = &.{
            "src/ittnotify/ittnotify_static.c",
            "src/ittnotify/jitprofiling.c",
        },
        .flags = &.{ "-DOV_BUILD_POSTFIX=\"\"", "-Wno-undef" },
    });

    // Keep OpenVINO and its GPU plugin in one C++ RTTI domain. Zig embeds its
    // libc++ statically in shared libraries, so passing ov::Any/std::vector
    // values across a separate plugin DSO fails even when the types match.
    for ([_][]const []const u8{
        &openvino_sources.openvino_intel_gpu_common_obj,
        &openvino_sources.openvino_intel_gpu_cpu_obj,
        &openvino_sources.openvino_intel_gpu_graph,
        &openvino_sources.openvino_intel_gpu_kernels,
        &openvino_sources.openvino_intel_gpu_ocl_obj,
        &openvino_sources.openvino_intel_gpu_ocl_v2_obj,
        &openvino_sources.openvino_intel_gpu_runtime,
    }) |files| addOpenVinoGroup(b, runtime_mod, ov, files, &openvino_gpu_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_intel_gpu_plugin, &openvino_gpu_static_plugin_flags);
    addOpenVinoGroup(b, runtime_mod, ov, &openvino_sources.openvino_intel_gpu_plugin_version, &openvino_gpu_static_plugin_version_flags);

    const runtime = b.addLibrary(.{
        .name = "openvino",
        // OpenVINO's C++ API passes ov::Any and STL objects across its API.
        // Zig statically embeds libc++, so a separate runtime DSO creates a
        // second RTTI domain. Archive these objects into the ORT provider DSO.
        .linkage = .static,
        .root_module = runtime_mod,
    });

    return .{ .runtime = runtime, .opencl = opencl };
}

const Parts = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    openvino: bool,
    ort: *std.Build.Dependency,
    protobuf: *std.Build.Dependency,
    onnx: *std.Build.Dependency,
    abseil: *std.Build.Dependency,
    re2: *std.Build.Dependency,
    cpuinfo: *std.Build.Dependency,
    json: *std.Build.Dependency,
    coremltools: ?*std.Build.Dependency,
    fp16: ?*std.Build.Dependency,
    protos: std.Build.LazyPath,
    coreml_protos: ?std.Build.LazyPath,
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
        const json = b.dependency("json", .{});
        const is_darwin = target.result.os.tag.isDarwin();
        const coremltools = if (is_darwin) b.dependency("coremltools", .{}) else null;
        const fp16 = if (is_darwin) b.dependency("fp16", .{}) else null;

        const config = b.addWriteFiles();
        _ = config.add("onnxruntime_config.h", b.fmt(ort_config_header, .{build_zon.version}));

        const protoc_mod = b.createModule(.{
            .target = b.graph.host,
            .optimize = optimize,
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

        var coreml_protos: ?std.Build.LazyPath = null;
        if (coremltools) |coreml| {
            const generate_coreml = b.addRunArtifact(protoc);
            for (sources.coreml_proto_sources) |proto| {
                generate_coreml.addFileArg(coreml.path(b.fmt("mlmodel/format/{s}", .{proto})));
            }
            generate_coreml.addArg("-I");
            generate_coreml.addDirectoryArg(coreml.path("mlmodel/format"));
            generate_coreml.addArg("--cpp_out");
            coreml_protos = generate_coreml.addOutputDirectoryArg("coreml_proto");
        }

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
            json.path("single_include"),
        }) catch @panic("OOM");

        const cxx_base = if (openvino) &ort_openvino_flags else &ort_flags;
        const cxx_flags = if (is_darwin)
            concatFlags(b, &.{ cxx_base, &.{
                "-gline-tables-only",
                "-DUSE_COREML=1",
                "-DCOREML_ENABLE_MLPROGRAM=1",
            } })
        else
            cxx_base;

        return .{
            .b = b,
            .target = target,
            .optimize = optimize,
            .openvino = openvino,
            .ort = ort,
            .protobuf = protobuf,
            .onnx = onnx,
            .abseil = abseil,
            .re2 = re2,
            .cpuinfo = cpuinfo,
            .json = json,
            .coremltools = coremltools,
            .fp16 = fp16,
            .protos = protos,
            .coreml_protos = coreml_protos,
            .includes = includes,
            .flags = cxx_flags,
        };
    }

    fn cxxModule(self: Parts, target: std.Build.ResolvedTarget) *std.Build.Module {
        const mod = self.b.createModule(.{
            .target = target,
            .optimize = self.optimize,
            .link_libc = true,
            .link_libcpp = true,
        });
        for (self.includes) |include| mod.addIncludePath(include);
        return mod;
    }

    fn linkCxx(_: Parts, _: *std.Build.Module) void {}

    fn runtime(self: Parts) *std.Build.Step.Compile {
        const b = self.b;
        const ort_root = self.ort.path("onnxruntime");
        const target_sources = sources.forTarget(self.target.result);

        const lib_mod = self.cxxModule(self.target);
        lib_mod.addIncludePath(self.protos);
        if (self.target.result.os.tag.isDarwin()) {
            lib_mod.linkFramework("Foundation", .{});
            lib_mod.linkFramework("CoreFoundation", .{});
            lib_mod.linkFramework("CoreML", .{});

            const coreml = self.coremltools.?;
            const coreml_protos = self.coreml_protos.?;
            lib_mod.addIncludePath(coreml_protos.dirname());
            lib_mod.addIncludePath(coreml.path(""));
            lib_mod.addIncludePath(coreml.path("mlmodel/src"));
            lib_mod.addIncludePath(coreml.path("modelpackage/src"));
            lib_mod.addIncludePath(self.json.path("single_include/nlohmann"));
            lib_mod.addIncludePath(self.fp16.?.path("include"));
            lib_mod.addCSourceFiles(.{
                .root = coreml_protos,
                .files = &sources.coreml_proto_generated_sources,
                .flags = self.flags,
            });
            lib_mod.addCSourceFiles(.{
                .root = ort_root,
                .files = &sources.ort_coreml_sources,
                .flags = self.flags,
            });
            lib_mod.addCSourceFiles(.{
                .root = b.path(""),
                .files = &sources.ort_coreml_override_sources,
                .flags = self.flags,
            });
            lib_mod.addCSourceFiles(.{
                .root = ort_root,
                .files = &sources.ort_coreml_objc_sources,
                .flags = concatFlags(b, &.{ self.flags, &.{"-fobjc-arc"} }),
            });
            lib_mod.addCSourceFiles(.{
                .root = coreml.path(""),
                .files = &sources.coremltools_sources,
                .flags = self.flags,
            });
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

const openvino_include_dirs = [_][]const u8{
    "src/common/conditional_compilation/include",
    "src/common/itt/include",
    "src/common/low_precision_transformations/include",
    "src/common/shutdown/include",
    "src/common/transformations/include",
    "src/common/transformations/src",
    "src/common/util/include",
    "src/core/dev_api",
    "src/core/include",
    "src/core/reference/include",
    "src/core/shape_inference/include",
    "src/core/src",
    "src/frontends/common/dev_api",
    "src/frontends/common/include",
    "src/frontends/common/src",
    "src/frontends/common_translators/include",
    "src/frontends/onnx/frontend/include",
    "src/frontends/paddle/include",
    "src/frontends/pytorch/include",
    "src/frontends/tensorflow/include",
    "src/frontends/tensorflow_lite/include",
    "src/inference/dev_api",
    "src/inference/include",
    "src/inference/src",
    "src/plugins/intel_gpu/include",
    "src/plugins/intel_gpu/src",
    "src/plugins/intel_gpu/src/graph",
    "src/plugins/intel_gpu/src/graph/impls",
    "src/plugins/intel_gpu/src/graph/include",
    "src/plugins/intel_gpu/src/kernel_selector",
    "src/plugins/intel_gpu/src/kernel_selector/kernels",
    "src/plugins/intel_gpu/src/runtime",
    "src/plugins/intel_gpu/thirdparty",
};

const openvino_common_flags = [_][]const u8{
    "-std=c++17",
    "-includeopenvino_itt_compat.hpp",
    "-DOpenVINO_VERSION=\"2026.3.0\"",
    "-DIN_OV_COMPONENT",
    "-DOV_BUILD_POSTFIX=\"\"",
    "-DOV_NATIVE_PARENT_PROJECT_ROOT_DIR=\"openvino-2026.3.0\"",
    "-DOV_THREAD=OV_THREAD_SEQ",
    // Keep Zig's C++ cache from reusing objects produced before the
    // OpenVINO ITT compatibility shim was applied to the source tree.
    "-DOV_ZIG_SOURCE_BUILD=3",
    "-fsigned-char",
    "-fno-sanitize=undefined",
    "-w",
};

const openvino_api_flags = [_][]const u8{"-DIMPLEMENT_OPENVINO_API"};
const openvino_core_internal_flags = [_][]const u8{"-DIN_OV_CORE_LIBRARY"};
const openvino_core_flags = openvino_api_flags ++ openvino_core_internal_flags ++ [_][]const u8{
    "-DXBYAK64",
    "-DXBYAK_NO_OP_NAMES",
};
const openvino_core_version_flags = openvino_core_flags ++ [_][]const u8{
    "-DCI_BUILD_NUMBER=\"2026.3.0-1-8a17657b995\"",
};
const openvino_frontend_flags = openvino_api_flags ++ [_][]const u8{
    "-DOPENVINO_STATIC_LIBRARY",
    "-DFRONTEND_LIB_PREFIX=\"libopenvino_\"",
    "-DFRONTEND_LIB_SUFFIX=\"_frontend.so.2630\"",
};
const openvino_onnx_frontend_flags = [_][]const u8{
    "-DONNX_ML=1",
    "-DONNX_NAMESPACE=onnx",
    "-DONNX_BUILD_SHARED=1",
    "-DONNX_OPSET_VERSION=24",
    "-Dget_front_end_data=get_front_end_data_onnx",
    "-Dget_api_version=get_api_version_onnx",
};
const openvino_runtime_flags = [_][]const u8{"-DIMPLEMENT_OPENVINO_RUNTIME_API"};
const openvino_reference_flags = openvino_core_internal_flags ++ [_][]const u8{
    "-DHAVE_AVX2",
    "-DXBYAK64",
    "-DXBYAK_NO_OP_NAMES",
};
const openvino_gpu_flags = [_][]const u8{
    "-DCL_TARGET_OPENCL_VERSION=300",
    "-DOV_GPU_OPENCL_HPP_HAS_BUS_INFO",
    "-DOV_GPU_OPENCL_HPP_HAS_UUID",
    "-DOV_GPU_USE_OPENCL_HPP",
    "-DOV_GPU_WITH_OCL_RT=1",
};
const openvino_gpu_plugin_flags = openvino_gpu_flags ++ [_][]const u8{
    "-DIMPLEMENT_OPENVINO_RUNTIME_PLUGIN",
};
const openvino_gpu_plugin_version_flags = openvino_gpu_plugin_flags ++ [_][]const u8{
    "-DCI_BUILD_NUMBER=\"2026.3.0-1-8a17657b995\"",
};
const openvino_gpu_static_plugin_flags = openvino_gpu_plugin_flags ++ [_][]const u8{
    "-DOV_CREATE_PLUGIN=create_plugin_engine_GPU",
};
const openvino_gpu_static_plugin_version_flags = openvino_gpu_static_plugin_flags ++ [_][]const u8{
    "-DCI_BUILD_NUMBER=\"2026.3.0-1-8a17657b995\"",
};

const openvino_plugins_header =
    \\#pragma once
    \\#include <map>
    \\#include <memory>
    \\#include <string>
    \\#define OPENVINO_STATIC_LIBRARY
    \\extern "C" void create_plugin_engine_GPU(std::shared_ptr<ov::IPlugin>&) noexcept(false);
    \\using CreatePluginEngineFunc = void(std::shared_ptr<ov::IPlugin>&);
    \\using CreateExtensionFunc = void(std::vector<ov::Extension::Ptr>&);
    \\struct Value {
    \\    CreatePluginEngineFunc* m_create_plugin_func;
    \\    CreateExtensionFunc* m_create_extensions_func;
    \\    std::map<std::string, std::string> m_default_config;
    \\};
    \\using Key = std::string;
    \\using PluginsStaticRegistry = std::map<Key, Value>;
    \\inline const std::map<Key, Value> get_compiled_plugins_registry() {
    \\    static const std::map<Key, Value> plugins = {
    \\        { "GPU", Value { create_plugin_engine_GPU, nullptr, {} } },
    \\    };
    \\    return plugins;
    \\}
;

const openvino_frontends_header =
    \\#pragma once
    \\#include "openvino/frontend/frontend.hpp"
    \\ov::frontend::FrontEndVersion get_api_version_onnx();
    \\void* get_front_end_data_onnx();
    \\namespace {
    \\using get_front_end_data_func = void*();
    \\using get_api_version_func = ov::frontend::FrontEndVersion();
    \\struct FrontendValue {
    \\    get_front_end_data_func* m_dataFunc;
    \\    get_api_version_func* m_versionFunc;
    \\};
    \\using FrontendsStaticRegistry = std::vector<FrontendValue>;
    \\const FrontendsStaticRegistry getStaticFrontendsRegistry() {
    \\    return { FrontendValue { get_front_end_data_onnx, get_api_version_onnx } };
    \\}
    \\}
;

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
