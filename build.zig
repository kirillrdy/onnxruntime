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
//!
//! `-Dopenvino=<prefix>` additionally builds the OpenVINO execution provider,
//! which is how ONNX Runtime reaches an Intel NPU. That build links a system
//! OpenVINO and so gives up the properties above; see `OpenVino`.

const std = @import("std");
const sources = @import("sources.zig");
const build_zon = @import("build.zig.zon");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const parts = Parts.init(b, target, optimize, OpenVino.option(b, target));
    b.installArtifact(parts.runtime());
    if (parts.openvino != null) {
        // Built once and passed along: the provider has to link the very
        // library that gets installed, or the ProviderHost pointer they are
        // supposed to share would sit in two different objects.
        const shared = parts.providersShared();
        b.installArtifact(shared);
        b.installArtifact(parts.openvinoProvider(shared));
    }
}

/// The ONNX Runtime static library, headers included. Linking it is all a
/// dependent has to do.
pub fn library(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    return Parts.init(b, target, optimize, null).runtime();
}

/// Put the C++ standard library a `-Dopenvino` build was compiled against on
/// a consuming module's link line.
///
/// Needed because that build runs on the host's libstdc++ rather than Zig's
/// libc++, and Zig does not carry a static library's own link objects across
/// to whoever links it -- so without this the final link ends in undefined
/// `__cxa_begin_catch` and friends. The default build needs nothing of the
/// sort; its libc++ comes along with the artifact.
///
///     const ort = b.dependency("onnxruntime", .{ ... });
///     exe.root_module.linkLibrary(ort.artifact("onnxruntime"));
///     @import("onnxruntime").linkStdCxx(b, exe.root_module);
pub fn linkStdCxx(b: *std.Build, mod: *std.Build.Module) void {
    Gnu.detect(b).link(mod);
}

/// A system OpenVINO to build the execution provider against.
///
/// The NPU is only reachable through Intel's own compiler, which ships as a
/// prebuilt binary inside the OpenVINO install, so this half of the build
/// cannot come from source the way the rest does. What it costs: the provider
/// links libopenvino.so, and because provider and runtime hand each other C++
/// objects across the dlopen boundary, the *whole* build has to match
/// OpenVINO's ABI -- GCC's libstdc++, with RTTI on, for the host only.
pub const OpenVino = struct {
    /// Where `openvino/openvino.hpp` lives. Defaults to `<prefix>/include`;
    /// package managers that split a build into `dev` and `lib` outputs need
    /// this pointed at the former.
    include: []const u8,
    /// Where `libopenvino.so` lives, along with `openvino/` or the plugins
    /// themselves -- the NPU plugin and Intel's NPU compiler among them.
    /// Defaults to `<prefix>/lib`; Intel's own archives put all of it in
    /// `runtime/lib/intel64`, which no prefix describes.
    lib: []const u8,

    fn option(b: *std.Build, target: std.Build.ResolvedTarget) ?OpenVino {
        // All three declared before any is read: an option only shows up in
        // `zig build --help`, and only becomes settable, once b.option has
        // been reached, so returning early would hide the rest.
        const prefix_option = b.option(
            []const u8,
            "openvino",
            "Prefix of a system OpenVINO (2026.0+) to build the execution provider against, for Intel NPU and GPU support",
        );
        const include_option = b.option(
            []const u8,
            "openvino-include",
            "Include directory of the OpenVINO headers, if not <prefix>/include",
        );
        const lib_option = b.option(
            []const u8,
            "openvino-lib",
            "Directory holding libopenvino.so and the plugins, if not <prefix>/lib",
        );

        const prefix = prefix_option orelse return null;
        const include = include_option orelse b.pathJoin(&.{ prefix, "include" });
        const lib = lib_option orelse b.pathJoin(&.{ prefix, "lib" });

        // The libstdc++ this drags in is the host's, found by asking the host
        // compiler. Handing those headers to a cross build would be nonsense.
        if (!target.query.isNative()) {
            std.log.err("-Dopenvino builds against the host's libstdc++ and cannot cross-compile", .{});
            std.process.exit(1);
        }
        return .{ .include = include, .lib = lib };
    }
};

/// Which C++ standard library the whole build runs on. Not a per-module
/// choice: every object here ends up in one process exchanging std::string
/// and std::vector, so they must agree.
const Cxx = union(enum) {
    /// Zig's own, statically linked. Depends on nothing but libc.
    libcxx,
    /// The host GCC's, to match a prebuilt libopenvino.so.
    libstdcxx: Gnu,
};

/// The host GCC installation, located by asking the compiler rather than by
/// guessing paths, so this works the same on a distro and inside a Nix store.
const Gnu = struct {
    /// libstdc++ header directories, in search order.
    include: []const []const u8,
    /// libstdc++ itself, and libgcc_s for the exception unwinder -- without
    /// the second, every throw is an undefined _Unwind_Resume. Full paths,
    /// because there is no way to ask for these by name: `stdc++` is matched
    /// by isLibCxxLibName in Module.linkSystemLibrary, again in the compiler
    /// driver, and both turn the request back into Zig's libc++ -- which then
    /// puts its own headers ahead of GCC's and breaks the build. `-l:file`
    /// syntax is no way out either; Zig resolves system libraries itself.
    ///
    /// libgcc_s is taken by its soname rather than the bare `.so`, which on
    /// some installs is a linker script pulling in a static `-lgcc` that Zig
    /// has no search path for.
    libraries: [2][]const u8,

    /// Put the libraries, and an RPATH to find them again at run time, on a
    /// module that is going to be linked into something.
    fn link(self: Gnu, mod: *std.Build.Module) void {
        for (self.libraries) |lib| {
            mod.addObjectFile(.{ .cwd_relative = lib });
            mod.addRPath(.{ .cwd_relative = std.fs.path.dirname(lib).? });
        }
    }

    fn detect(b: *std.Build) Gnu {
        const cxx = b.graph.environ_map.get("CXX") orelse "c++";

        // The include search list is printed on stderr, between two markers,
        // hence the redirect: runAllowFail only hands back stdout. Only the
        // C++ directories are wanted -- Zig supplies libc itself.
        const verbose = run(b, b.fmt("{s} -E -x c++ /dev/null -v 2>&1", .{cxx}));
        var include: std.ArrayList([]const u8) = .empty;
        var lines = std.mem.splitScalar(u8, verbose, '\n');
        var listing = false;
        while (lines.next()) |line| {
            const dir = std.mem.trim(u8, line, " \t\r");
            if (std.mem.startsWith(u8, dir, "#include <...>")) listing = true;
            if (std.mem.startsWith(u8, dir, "End of search list")) break;
            if (listing and std.mem.indexOf(u8, dir, "/c++/") != null) {
                include.append(b.allocator, b.dupe(dir)) catch @panic("OOM");
            }
        }
        if (include.items.len == 0) {
            std.log.err("{s} reported no libstdc++ include directories", .{cxx});
            std.process.exit(1);
        }

        // -print-file-name echoes the name back unchanged when it cannot
        // resolve it, which is the only signal that the library is missing.
        var libraries: [2][]const u8 = undefined;
        for ([_][]const u8{ "libstdc++.so", "libgcc_s.so.1" }, &libraries) |file, *out| {
            const path = std.mem.trim(u8, run(b, b.fmt("{s} -print-file-name={s}", .{ cxx, file })), " \t\r\n");
            if (std.mem.eql(u8, path, file)) {
                std.log.err("{s} could not locate {s}", .{ cxx, file });
                std.process.exit(1);
            }
            out.* = b.dupe(path);
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

/// The dependencies, include set and generated headers that both the runtime
/// and the OpenVINO provider are built from. `b.dependency` is memoised, so
/// naming them once here is only about not repeating the list.
const Parts = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    openvino: ?OpenVino,
    cxx: Cxx,
    ort: std.Build.LazyPath,
    ort_root: std.Build.LazyPath,
    protobuf: std.Build.LazyPath,
    onnx: std.Build.LazyPath,
    abseil: std.Build.LazyPath,
    re2: std.Build.LazyPath,
    cpuinfo: std.Build.LazyPath,
    protos: std.Build.LazyPath,
    includes: []const std.Build.LazyPath,

    fn init(
        b: *std.Build,
        target: std.Build.ResolvedTarget,
        optimize: std.builtin.OptimizeMode,
        openvino: ?OpenVino,
    ) Parts {
        const ort = b.dependency("ort_src", .{}).path(".");
        const protobuf = b.dependency("protobuf", .{}).path(".");
        const onnx = b.dependency("onnx", .{}).path(".");
        const abseil = b.dependency("abseil", .{}).path(".");
        const re2 = b.dependency("re2", .{}).path(".");
        const cpuinfo = b.dependency("cpuinfo", .{}).path(".");

        const config = b.addWriteFiles();
        _ = config.add("onnxruntime_config.h", b.fmt(ort_config_header, .{build_zon.version}));

        // protoc runs during the build, so it is built for the host even when
        // the runtime itself is cross-compiled.
        const protos = generateOnnxProto(b, buildProtoc(b, protobuf), onnx);

        // Deliberately excludes `protos`. A generated include path is a step
        // dependency, so listing it here would park all of mlas behind protoc
        // instead of letting the two run at once; only the runtime proper needs
        // the generated headers, and it adds them for itself below.
        const includes = b.allocator.dupe(std.Build.LazyPath, &.{
            config.getDirectory(),
            ort.path(b, "include/onnxruntime"),
            ort.path(b, "include/onnxruntime/core/session"),
            ort.path(b, "onnxruntime"),
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
        }) catch @panic("OOM");

        return .{
            .b = b,
            .target = target,
            .optimize = optimize,
            .openvino = openvino,
            .cxx = if (openvino == null) .libcxx else .{ .libstdcxx = Gnu.detect(b) },
            .ort = ort,
            .ort_root = ort.path(b, "onnxruntime"),
            .protobuf = protobuf,
            .onnx = onnx,
            .abseil = abseil,
            .re2 = re2,
            .cpuinfo = cpuinfo,
            .protos = protos,
            .includes = includes,
        };
    }

    /// A module compiled as C++ against `self.cxx`, seeded with the shared
    /// include set. Every module here wants the same libc/libc++ treatment;
    /// only the resolved target and any extra includes differ.
    fn cxxModule(self: Parts, target: std.Build.ResolvedTarget) *std.Build.Module {
        const mod = self.b.createModule(.{
            .target = target,
            .optimize = self.optimize,
            .link_libc = true,
            .link_libcpp = self.cxx == .libcxx,
        });
        for (self.includes) |include| mod.addIncludePath(include);
        switch (self.cxx) {
            .libcxx => {},
            .libstdcxx => |gnu| {
                // -I, not -isystem: Zig lists its own libc directories ahead of
                // every -isystem path, and libstdc++'s <cstdlib> reaches its
                // libc counterpart by #include_next, which only searches what
                // comes after. The C++ headers have to be found first for that
                // to land on glibc's stdlib.h.
                for (gnu.include) |dir| mod.addIncludePath(.{ .cwd_relative = dir });
                // Headers only. The libraries themselves are added by
                // `linkStdCxx`, and only to things that actually link: an
                // object file handed to a module that becomes a static
                // archive is stored *in* the archive, where it is not an
                // object the linker can use, and ld.lld's complaint about
                // that counts as a failed build.
            },
        }
        return mod;
    }

    /// Add the C++ standard library to a module that is about to be linked
    /// into a shared library or an executable. `cxxModule` deliberately does
    /// not do this: most modules here end up as static archives, which is the
    /// one place these files must not go.
    fn linkCxx(self: Parts, mod: *std.Build.Module) void {
        switch (self.cxx) {
            .libcxx => {}, // Zig links its own.
            .libstdcxx => |gnu| gnu.link(mod),
        }
    }

    /// Flags for every C++ translation unit in the build.
    fn flags(self: Parts) []const []const u8 {
        return switch (self.cxx) {
            .libcxx => &ort_flags,
            // OpenVINO's headers throw and catch by type, and its own build
            // has RTTI on, so the no-RTTI pair has to come back out.
            //
            // -Wno-invalid-constexpr: glibc 2.41 dropped
            // __CORRECT_ISO_CPP_MATH_H_PROTO, so libstdc++'s <cmath> now
            // declares its own constexpr acos(float) and friends over
            // __builtin_acosf. GCC folds those at compile time; clang does
            // not, and rejects the definitions outright rather than warning.
            .libstdcxx => concatFlags(self.b, &.{
                withoutFlags(self.b, &ort_flags, &.{ "-fno-rtti", "-DORT_NO_RTTI" }),
                &.{"-Wno-invalid-constexpr"},
            }),
        };
    }

    fn runtime(self: Parts) *std.Build.Step.Compile {
        const b = self.b;
        const ort_flags_ = self.flags();

        const lib_mod = self.cxxModule(self.target);
        lib_mod.addIncludePath(self.protos);

        lib_mod.addCSourceFiles(.{ .root = self.protos, .files = &sources.onnx_proto_sources, .flags = ort_flags_ });
        lib_mod.addCSourceFiles(.{
            .root = self.onnx,
            .files = &sources.onnx_sources,
            .flags = concatFlags(b, &.{ ort_flags_, &.{"-D__ONNX_DISABLE_STATIC_REGISTRATION"} }),
        });
        lib_mod.addCSourceFiles(.{ .root = self.abseil, .files = &sources.abseil_sources, .flags = ort_flags_ });
        lib_mod.addCSourceFiles(.{ .root = self.re2, .files = &sources.re2_sources, .flags = ort_flags_ });
        lib_mod.addCSourceFiles(.{ .root = self.protobuf, .files = &sources.protobuf_lite_sources, .flags = ort_flags_ });
        lib_mod.addCSourceFiles(.{ .root = self.cpuinfo, .files = &sources.cpuinfo_sources, .flags = &ort_c_flags });
        lib_mod.addCSourceFiles(.{
            .root = self.ort.path(b, "model_package"),
            .files = &sources.model_package_sources,
            .flags = ort_flags_,
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
            lib_mod.addCSourceFiles(.{ .root = self.ort_root, .files = list, .flags = ort_flags_ });
        }
        for (sources.ort_file_flags) |override| {
            lib_mod.addCSourceFiles(.{
                .root = self.ort_root,
                .files = &.{override.file},
                .flags = concatFlags(b, &.{ ort_flags_, override.flags }),
            });
        }

        const lib = b.addLibrary(.{
            .name = "onnxruntime",
            .linkage = .static,
            .root_module = lib_mod,
        });

        for (sources.ort_mlas_groups, 0..) |group, index| {
            var query = self.target.query;
            for (group.features) |feature| query.cpu_features_add.addFeature(@intFromEnum(feature));

            const group_mod = self.cxxModule(b.resolveTargetQuery(query));
            group_mod.addCSourceFiles(.{
                .root = self.ort_root,
                .files = group.files,
                .flags = concatFlags(b, &.{ ort_flags_, &mlas_group_flags, group.flags }),
            });
            lib.root_module.linkLibrary(b.addLibrary(.{
                .name = b.fmt("onnxruntime-mlas-{d}", .{index}),
                .linkage = .static,
                .root_module = group_mod,
            }));
        }

        // Travels with the artifact: a module that links this library gets the C
        // API headers on its include path without naming a path itself.
        lib.installHeadersDirectory(self.ort.path(b, "include/onnxruntime/core/session"), "", .{});
        return lib;
    }

    /// libonnxruntime_providers_shared.so: one global ProviderHost pointer.
    ///
    /// Small, and shared for exactly one reason. The runtime dlopens this file
    /// -- by name, out of the directory the running binary sits in -- and
    /// calls Provider_SetHost to leave a vtable of its internals in the
    /// global. Each provider .so links the same file and calls
    /// Provider_GetHost to pick the vtable back up. A static library could not
    /// do that job: each side would get its own copy of the global.
    ///
    /// The consequence for anyone linking the static runtime is that this file
    /// has to be installed next to their executable, or loading a provider
    /// fails with `undefined symbol: Provider_GetHost`.
    fn providersShared(self: Parts) *std.Build.Step.Compile {
        const b = self.b;
        const mod = self.cxxModule(self.target);
        mod.addCSourceFiles(.{
            .root = self.ort_root,
            .files = &sources.ort_provider_host_sources,
            .flags = self.flags(),
        });
        self.linkCxx(mod);
        const shared = b.addLibrary(.{
            .name = "onnxruntime_providers_shared",
            .linkage = .dynamic,
            .root_module = mod,
        });
        shared.setVersionScript(self.ort_root.path(b, "core/providers/shared/version_script.lds"));
        return shared;
    }

    /// libonnxruntime_providers_openvino.so.
    ///
    /// A shared library and not part of libonnxruntime.a, because that is how
    /// the runtime expects to find an OpenVINO EP: it dlopens the file and
    /// calls CreateEpFactories. What it needs back from the runtime it reaches
    /// through `providersShared`, which it links.
    fn openvinoProvider(self: Parts, shared: *std.Build.Step.Compile) *std.Build.Step.Compile {
        const b = self.b;
        const openvino = self.openvino.?;

        const mod = self.cxxModule(self.target);
        mod.addIncludePath(self.protos);
        mod.addIncludePath(.{ .cwd_relative = openvino.include });

        const flags_ = concatFlags(b, &.{
            self.flags(),
            &.{
                // OpenVINO 2024.4 and up let the provider allocate NPU-visible
                // memory itself rather than copying into it.
                "-DUSE_OVEP_NPU_MEMORY=1",
                "-DFILE_NAME=\"libonnxruntime_providers_openvino.so\"",
                // Upstream only ever compiles this provider with GCC, and two
                // of its habits are hard errors for clang rather than the
                // warnings GCC settles for: `enum class type` as an elaborated
                // parameter type in exceptions.h, and a long narrowed to
                // uint32_t inside a braced initializer in basic_backend.h.
                "-Wno-elaborated-enum-class",
                "-Wno-c++11-narrowing",
            },
        });
        mod.addCSourceFiles(.{ .root = self.ort_root, .files = &sources.ort_openvino_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = self.ort_root, .files = &sources.ort_provider_shared_sources, .flags = flags_ });

        // The provider carries its own onnx protobuf and abseil rather than
        // sharing the runtime's: the version script below hides every symbol
        // that is not an entry point, so the two copies cannot collide.
        mod.addCSourceFiles(.{ .root = self.protos, .files = &sources.onnx_proto_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = self.protobuf, .files = &sources.protobuf_lite_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = self.abseil, .files = &sources.abseil_sources, .flags = flags_ });

        mod.addLibraryPath(.{ .cwd_relative = openvino.lib });
        mod.linkSystemLibrary("openvino", .{});
        // Loaded at run time from where it was linked, so the plugins beside
        // it -- the NPU plugin, and Intel's NPU compiler -- are found too.
        mod.addRPath(.{ .cwd_relative = openvino.lib });
        self.linkCxx(mod);

        const provider = b.addLibrary(.{
            .name = "onnxruntime_providers_openvino",
            .linkage = .dynamic,
            .root_module = mod,
        });
        // Exports GetProvider, CreateEpFactories and ReleaseEpFactory, and
        // hides the rest.
        provider.setVersionScript(self.ort_root.path(b, "core/providers/openvino/version_script.lds"));
        provider.root_module.linkLibrary(shared);
        // The runtime dlopens providers_shared with RTLD_GLOBAL before any
        // provider, so by this point the symbol is already in the process.
        // $ORIGIN covers the case where it is not: the two are installed
        // side by side.
        provider.root_module.addRPath(.{ .cwd_relative = "$ORIGIN" });
        return provider;
    }
};

fn buildProtoc(b: *std.Build, protobuf: std.Build.LazyPath) *std.Build.Step.Compile {
    // Pinned rather than inherited from the caller: this protoc runs three
    // times on three small .proto files, so codegen quality buys nothing, and
    // -Os builds protobuf roughly twice as fast as any other mode. Pinning
    // also keeps the whole 167-file host build out of the consumer's
    // -Doptimize cache key, so it is paid once per host instead of per mode.
    //
    // Always Zig's libc++, even in an OpenVINO build: protoc is a build-time
    // tool that outputs .cc files, and shares an address space with nothing.
    const protoc_mod = b.createModule(.{
        .target = b.graph.host,
        .optimize = .ReleaseSmall,
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

fn concatFlags(b: *std.Build, parts: []const []const []const u8) []const []const u8 {
    return std.mem.concat(b.allocator, []const u8, parts) catch @panic("OOM");
}

fn withoutFlags(b: *std.Build, flags: []const []const u8, remove: []const []const u8) []const []const u8 {
    var kept: std.ArrayList([]const u8) = .empty;
    outer: for (flags) |flag| {
        for (remove) |drop| if (std.mem.eql(u8, flag, drop)) continue :outer;
        kept.append(b.allocator, flag) catch @panic("OOM");
    }
    return kept.items;
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
