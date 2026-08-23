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
//! which is how ONNX Runtime reaches an Intel NPU, and `-Dcuda=<prefix>` builds
//! the CUDA one, which is how it reaches an NVIDIA GPU. Each links a system
//! library -- and CUDA additionally needs nvcc, since nothing else emits device
//! code -- so either gives up the properties above; see `OpenVino` and `Cuda`.

const std = @import("std");
const sources = @import("sources.zig");
const build_zon = @import("build.zig.zon");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const fetch = fetchStep(b);

    const parts = Parts.init(b, target, optimize, OpenVino.option(b, target, fetch), Cuda.option(b, target));
    const lib = parts.runtime();
    b.installArtifact(lib);

    // The Zig side of the package: `onnx.zig` over the library just built.
    // Linking the artifact into the module is what carries the C headers along,
    // so the `@cInclude` at the top of onnx.zig resolves, and what makes a
    // consumer that imports this module link the runtime without asking.
    const mod = b.addModule("onnxruntime", .{
        .root_source_file = b.path("onnx.zig"),
        .target = target,
        .optimize = optimize,
    });
    mod.linkLibrary(lib);
    // A provider build runs on GCC's libstdc++, and Zig does not carry a static
    // archive's own link objects across to whoever links it. Putting them on
    // the module means they reach every executable that imports it.
    parts.linkCxx(mod);

    if (parts.openvino != null or parts.cuda != null) {
        // Built once and passed along: a provider has to link the very library
        // that gets installed, or the ProviderHost pointer they are supposed to
        // share would sit in two different objects. One `shared` serves both
        // providers for the same reason.
        const shared = parts.providersShared();
        b.installArtifact(shared);
        if (parts.openvino != null) b.installArtifact(parts.openvinoProvider(shared));
        if (parts.cuda != null) b.installArtifact(parts.cudaProvider(shared));
    }
}

/// The ONNX Runtime static library, headers included. Linking it is all a
/// dependent has to do.
pub fn library(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
) *std.Build.Step.Compile {
    return Parts.init(b, target, optimize, null, null).runtime();
}

/// Which execution provider a build ended up carrying.
pub const Provider = enum {
    /// None beyond the CPU one compiled into the runtime, and so nothing to
    /// install and no system library behind it.
    none,
    openvino,
    cuda,

    /// The artifact's name, which is also the file name once installed.
    fn artifactName(self: Provider) []const u8 {
        return switch (self) {
            .none => unreachable,
            .openvino => "onnxruntime_providers_openvino",
            .cuda => "onnxruntime_providers_cuda",
        };
    }
};

/// What `select` hands back: the dependency, which provider it carries, and the
/// few things a consumer has to do differently because of that choice.
///
/// The point of it is that those differences are this package's business, not
/// its consumers'. Which C++ standard library the build runs on, whether the
/// executable has to export its dynamic symbols, where `providers_shared` is
/// installed and why -- all of it follows from how the provider is built, and
/// none of it is knowable from outside without repeating the reasoning.
pub const Selection = struct {
    b: *std.Build,
    dependency: *std.Build.Dependency,
    provider: Provider,
    /// The compiler whose libstdc++ this build runs on, or null when it runs on
    /// Zig's own libc++ and needs nothing added.
    cxx: ?[]const u8,
    /// Set when the OpenVINO behind this was downloaded rather than named, and
    /// so sits in the build cache with its own oneTBB beside it, neither of
    /// which is anywhere the dynamic loader looks. See `runtimeLibraryPaths`.
    fetched_openvino: bool = false,

    /// Directories a program built against this has to have on its library path
    /// at run time, beyond what the system already provides. Empty unless the
    /// OpenVINO was downloaded: an installed one is the system's problem, and
    /// the CUDA provider carries an rpath to its toolkit.
    pub fn runtimeLibraryPaths(self: Selection) []const []const u8 {
        return if (self.fetched_openvino) openvinoRuntimeLibraryPaths(self.b) else &.{};
    }

    /// The Zig module wrapping the C API -- `onnx.zig`, with the runtime it
    /// calls into already on its link line.
    pub fn module(self: Selection) *std.Build.Module {
        return self.dependency.module("onnxruntime");
    }

    /// Add that module to `mod`'s imports, under the name "onnxruntime".
    /// Linking the runtime comes with it, since the module carries it.
    pub fn addImport(self: Selection, mod: *std.Build.Module) void {
        mod.addImport("onnxruntime", self.module());
    }

    /// What an executable needs beyond `link`.
    ///
    /// The standard library again, because Zig does not carry a static
    /// archive's own link objects across to whoever links it. And, for CUDA,
    /// `rdynamic`: some of its kernels subclass their CPU counterparts, so the
    /// provider is dlopened with an undefined reference to
    /// `onnxruntime::Einsum::DeviceCompute` still outstanding. Upstream ships
    /// the runtime as a shared library and that resolves against it; here the
    /// runtime is a static archive inside the executable, and an executable
    /// exports nothing to the libraries it loads unless it is asked to.
    pub fn linkExecutable(self: Selection, exe: *std.Build.Step.Compile) void {
        if (self.cxx) |cxx| linkStdCxx(self.b, exe.root_module, cxx);
        if (self.provider == .cuda) exe.rdynamic = true;
    }

    /// Install the provider library and the bridge that reaches it, and answer
    /// with the path the provider landed at -- which is what the program has to
    /// be told, since the runtime dlopens it by path. Empty for `.none`.
    ///
    /// `libonnxruntime_providers_shared.so` goes to `.bin` rather than the
    /// usual `.lib` because the runtime dlopens it *by name*, out of the
    /// directory the running binary sits in.
    pub fn installProvider(self: Selection) []const u8 {
        const b = self.b;
        if (self.provider == .none) return "";

        b.getInstallStep().dependOn(&b.addInstallArtifact(
            self.dependency.artifact("onnxruntime_providers_shared"),
            .{ .dest_dir = .{ .override = .bin } },
        ).step);

        const name = self.provider.artifactName();
        const install = b.addInstallArtifact(self.dependency.artifact(name), .{});
        b.getInstallStep().dependOn(&install.step);
        return b.getInstallPath(.lib, b.fmt("lib{s}.so", .{name}));
    }
};

/// Declare this package's provider options on `b`, resolve the dependency they
/// select, and return it along with what the choice implies.
///
/// A consumer that wants an execution provider should reach for this rather
/// than `b.dependency` directly: the options, their defaults and their help
/// text are declared once, here, instead of being restated by every build that
/// forwards them -- where they would drift.
pub fn select(
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    fallback: Provider,
) Selection {
    // Declared here rather than resolved through `OpenVino.option`, which is
    // what this package's own build() calls. That returns a struct already
    // worked out against the machine -- and, for a fetched OpenVINO, a download
    // step belonging to whichever builder asked for it. A consumer is
    // forwarding, so what it wants is the options as they were typed.
    //
    // All seven asked for before any is read: an option only shows up in `zig
    // build --help`, and only becomes settable, once b.option has been reached.
    const openvino = b.option(
        []const u8,
        "openvino",
        "Prefix of a system OpenVINO (2026.0+) to build the execution provider against, for Intel NPU and GPU support",
    );
    const openvino_include = b.option(
        []const u8,
        "openvino-include",
        "Include directory of the OpenVINO headers, if not <prefix>/include",
    );
    const openvino_lib = b.option(
        []const u8,
        "openvino-lib",
        "Directory holding libopenvino.so and the plugins, if not <prefix>/lib",
    );
    const cuda = b.option(
        []const u8,
        "cuda",
        "Prefix of a CUDA toolkit (12.x) to build the execution provider against, for NVIDIA GPU support",
    );
    const cudnn_include = b.option(
        []const u8,
        "cudnn-include",
        "Directory holding cudnn.h, if not <cuda prefix>/include",
    );
    const cuda_arch = b.option(
        []const u8,
        "cuda-arch",
        "Compute capability to emit device code for (default: 89, Ada -- RTX 40 series)",
    );
    const cuda_ccbin = b.option(
        []const u8,
        "cuda-ccbin",
        "Host C++ compiler for nvcc to drive, if the default c++ is newer than the toolkit accepts",
    );

    // One runtime takes one provider library. Two would need two of everything
    // in `Selection` -- and the device asked for would stop naming one of them.
    if (openvino != null and cuda != null) {
        std.log.err("-Dopenvino and -Dcuda cannot be combined: pick the one device to build for", .{});
        std.process.exit(1);
    }

    // Naming a prefix is always honoured. Otherwise `fallback` decides, which
    // is how a consumer's own device option reaches down here: asking for a
    // device the CPU provider cannot serve is what calls for a provider.
    const provider: Provider = if (openvino != null)
        .openvino
    else if (cuda != null)
        .cuda
    else
        fallback;

    switch (provider) {
        .none => return .{
            .b = b,
            .dependency = b.dependency("onnxruntime", .{ .target = target, .optimize = optimize }),
            .provider = .none,
            .cxx = null,
        },

        .openvino => return .{
            .b = b,
            // With no prefix named there is nothing installed to build
            // against, so the package downloads Intel's release instead.
            // -Dopenvino-include and -Dopenvino-lib are for describing an
            // installed one whose layout is unusual -- a split-output package
            // manager, say -- so they ride along with the prefix and not with
            // the download, whose layout the package already knows.
            .dependency = if (openvino) |prefix| b.dependency("onnxruntime", .{
                .target = target,
                .optimize = optimize,
                .openvino = prefix,
                .@"openvino-include" = openvino_include orelse b.pathJoin(&.{ prefix, "include" }),
                .@"openvino-lib" = openvino_lib orelse b.pathJoin(&.{ prefix, "lib" }),
            }) else b.dependency("onnxruntime", .{
                .target = target,
                .optimize = optimize,
                .@"openvino-fetch" = true,
            }),
            .provider = .openvino,
            .cxx = b.graph.environ_map.get("CXX") orelse "c++",
            .fetched_openvino = openvino == null,
        },

        .cuda => {
            // No equivalent of -Dopenvino-fetch: NVIDIA's toolkit is not a
            // tarball to unpack beside the build, and nvcc has to come out of a
            // real installation.
            const prefix = cuda orelse {
                std.log.err(
                    "reaching an NVIDIA GPU needs -Dcuda=<prefix>: there is no toolkit to download, and nvcc is what compiles the provider's device code",
                    .{},
                );
                std.process.exit(1);
            };
            const ccbin = cuda_ccbin orelse b.graph.environ_map.get("CXX") orelse "c++";
            return .{
                .b = b,
                .dependency = b.dependency("onnxruntime", .{
                    .target = target,
                    .optimize = optimize,
                    .cuda = prefix,
                    .@"cudnn-include" = cudnn_include orelse b.pathJoin(&.{ prefix, "include" }),
                    .@"cuda-arch" = cuda_arch orelse "89",
                    .@"cuda-ccbin" = ccbin,
                }),
                .provider = .cuda,
                // Whatever nvcc drives, since the .cu objects and the .cc
                // objects have to agree -- see `Cuda`.
                .cxx = ccbin,
            };
        },
    }
}

/// Put the C++ standard library a `-Dopenvino` or `-Dcuda` build was compiled
/// against on a consuming module's link line.
///
/// Needed because those builds run on GCC's libstdc++ rather than Zig's libc++,
/// and Zig does not carry a static library's own link objects across to whoever
/// links it -- so without this the final link ends in undefined
/// `__cxa_begin_catch` and friends. The default build needs nothing of the
/// sort; its libc++ comes along with the artifact.
///
/// `cxx` names the compiler to take libstdc++ from, and must be the same one
/// the dependency was configured with -- for a -Dcuda build, whatever was
/// passed as -Dcuda-ccbin. Null asks the host default, which is right for
/// -Dopenvino and for a CUDA toolkit new enough to accept it.
///
///     const ort = b.dependency("onnxruntime", .{ ... });
///     exe.root_module.linkLibrary(ort.artifact("onnxruntime"));
///     @import("onnxruntime").linkStdCxx(b, exe.root_module, null);
pub fn linkStdCxx(b: *std.Build, mod: *std.Build.Module, cxx: ?[]const u8) void {
    Gnu.detect(b, cxx orelse b.graph.environ_map.get("CXX") orelse "c++").link(mod);
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
    /// The download this was unpacked from, for the provider to wait on, or
    /// null when an installed OpenVINO was named and there is nothing to fetch.
    fetch: ?*std.Build.Step.Run,

    fn option(b: *std.Build, target: std.Build.ResolvedTarget, fetch: *std.Build.Step.Run) ?OpenVino {
        // All four declared before any is read: an option only shows up in
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
        const fetch_option = b.option(
            bool,
            "openvino-fetch",
            "Build against the OpenVINO release `fetch-openvino` downloads, rather than one installed on the machine",
        ) orelse false;

        if (prefix_option == null and !fetch_option) return null;
        if (prefix_option != null and fetch_option) {
            std.log.err("-Dopenvino names an installed OpenVINO and -Dopenvino-fetch downloads one; pass one or the other", .{});
            std.process.exit(1);
        }

        // The libstdc++ this drags in is the host's, found by asking the host
        // compiler. Handing those headers to a cross build would be nonsense.
        if (!target.query.isNative()) {
            std.log.err("an OpenVINO build runs on the host's libstdc++ and cannot cross-compile", .{});
            std.process.exit(1);
        }

        const default_include, const default_lib, const fetch_step = if (prefix_option) |prefix|
            .{ b.pathJoin(&.{ prefix, "include" }), b.pathJoin(&.{ prefix, "lib" }), null }
        else
            .{ b.pathJoin(&.{ openvinoRoot(b), "runtime", "include" }), b.pathJoin(&.{ openvinoRoot(b), "runtime", "lib", "intel64" }), fetch };

        return .{
            .include = include_option orelse default_include,
            .lib = lib_option orelse default_lib,
            .fetch = fetch_step,
        };
    }
};

/// A CUDA toolkit to build the execution provider against, for an NVIDIA GPU.
///
/// Device code cannot come from source the way the rest of this build does:
/// only NVIDIA's own nvcc emits it, and nvcc drives a *host* compiler for the
/// C++ half of every `.cu` file. That host compiler is GCC, so -Dcuda pulls the
/// whole build onto GCC's libstdc++ for exactly the reason `OpenVino` does --
/// provider and runtime pass each other C++ objects, so one ABI has to win.
pub const Cuda = struct {
    /// Prefix holding `bin/nvcc`, `include/cuda_runtime.h` and `lib/libcudart.so`.
    prefix: []const u8,
    /// Where `cudnn.h` lives. cuDNN is dlopened by name at run time and never
    /// linked, so this is headers only -- but the provider will not compile
    /// without them.
    cudnn_include: []const u8,
    /// The compute capability to emit device code for, e.g. "89" for Ada. One
    /// architecture, not a fat binary: each one compiled costs another pass
    /// over every kernel, and a build this size is slow enough already.
    arch: []const u8,
    /// Host compiler nvcc passes to its `-ccbin` flag. nvcc runs this compiler
    /// over every host translation unit inside a `.cu` file, so the runtime
    /// and the other providers have to link whatever libstdc++ it supplies.
    ccbin: []const u8,

    fn compiler(self: Cuda, b: *std.Build) []const u8 {
        return b.pathJoin(&.{ self.prefix, "bin", "nvcc" });
    }

    /// Read the toolkit's version as an integer like `1208` for 12.8, probed
    /// from nvcc.
    fn version(self: Cuda, b: *std.Build) u32 {
        var code: u8 = undefined;
        const out = b.runAllowFail(&.{ self.compiler(b), "--version" }, &code, .ignore) catch |err| {
            std.log.err("running `{s} --version`: {s}", .{ self.compiler(b), @errorName(err) });
            std.process.exit(1);
        };
        // nvcc: NVIDIA (R) Cuda compiler driver
        // Copyright (c) 2005-2024 NVIDIA Corporation
        // Built on Thu_Sep_12_02:18:05_PDT_2024
        // Cuda compilation tools, release 12.6, V12.6.77
        const release_marker = "release ";
        const pos = std.mem.indexOf(u8, out, release_marker) orelse return 0;
        const rest = out[pos + release_marker.len ..];
        var parts = std.mem.splitScalar(u8, rest, '.');
        const major = std.fmt.parseInt(u32, parts.next() orelse return 0, 10) catch return 0;
        const minor_str = parts.next() orelse return 0;
        const comma = std.mem.indexOfScalar(u8, minor_str, ',') orelse minor_str.len;
        const minor = std.fmt.parseInt(u32, minor_str[0..comma], 10) catch return 0;
        return major * 100 + minor;
    }

    fn option(b: *std.Build, target: std.Build.ResolvedTarget) ?Cuda {
        // All four declared before any is read: an option only shows up in
        // `zig build --help`, and only becomes settable, once b.option has
        // been reached.
        const prefix_option = b.option(
            []const u8,
            "cuda",
            "Prefix of a CUDA toolkit (12.x) to build the execution provider against, for NVIDIA GPU support",
        );
        const cudnn_include_option = b.option(
            []const u8,
            "cudnn-include",
            "Directory holding cudnn.h, if not <prefix>/include",
        );
        const arch_option = b.option(
            []const u8,
            "cuda-arch",
            "Compute capability to emit device code for (default: 89, Ada -- RTX 40 series)",
        );
        const ccbin_option = b.option(
            []const u8,
            "cuda-ccbin",
            "Host C++ compiler for nvcc to drive, if the default c++ is newer than the toolkit accepts",
        );

        const prefix = prefix_option orelse return null;
        if (!target.query.isNative()) {
            std.log.err("-Dcuda builds against the host's libstdc++ and cannot cross-compile", .{});
            std.process.exit(1);
        }
        return .{
            .prefix = prefix,
            .cudnn_include = cudnn_include_option orelse b.pathJoin(&.{ prefix, "include" }),
            .arch = arch_option orelse "89",
            .ccbin = ccbin_option orelse b.graph.environ_map.get("CXX") orelse "c++",
        };
    }
};

/// The OpenVINO release `fetch-openvino` downloads.
///
/// Pinned to a build rather than a version: Intel's archive names itself by
/// commit, and the NPU compiler inside it is what decides whether a graph
/// compiles at all, so "2026.3" on its own is not a thing to depend on. The
/// checksum is the one Intel publishes beside the archive.
const openvino_version = "2026.3.0";
const openvino_build = "22451.bd8d6542e3c";
const openvino_archive = "openvino_toolkit_ubuntu24_" ++ openvino_version ++ "." ++ openvino_build ++ "_x86_64.tgz";
const openvino_url = "https://storage.openvinotoolkit.org/repositories/openvino/packages/2026.3/linux/" ++ openvino_archive;
const openvino_sha256 = "0fa43c270bb6bc17bcf8c2c5fc5aa4595377292d54ef38084d8a0390daf4af98";

/// Where the archive is unpacked: the *consumer's* build cache, since a
/// dependency inherits its parent's `cache_root`. So one copy is shared by
/// every configuration of a project, and `zig build --clean` reaches it.
fn openvinoRoot(b: *std.Build) []const u8 {
    return b.cache_root.join(b.allocator, &.{ "onnxruntime", "openvino" }) catch @panic("OOM");
}

/// Directories a program built against the downloaded OpenVINO has to have on
/// its library path at run time.
///
/// The provider carries an rpath to the library directory, but `libopenvino.so`
/// itself carries none, and it links oneTBB -- which is in the archive, in a
/// directory of its own, and not anywhere the dynamic loader looks. Naming both
/// here rather than only oneTBB keeps this true of a program run from any
/// working directory.
pub fn openvinoRuntimeLibraryPaths(b: *std.Build) []const []const u8 {
    const root = openvinoRoot(b);
    const paths = b.allocator.alloc([]const u8, 2) catch @panic("OOM");
    paths[0] = b.pathJoin(&.{ root, "runtime", "lib", "intel64" });
    paths[1] = b.pathJoin(&.{ root, "runtime", "3rdparty", "tbb", "lib" });
    return paths;
}

/// `zig build fetch-openvino`, and the step the provider hangs off.
fn fetchStep(b: *std.Build) *std.Build.Step.Run {
    const fetch_exe = b.addExecutable(.{
        .name = "fetch",
        .root_module = b.createModule(.{
            .root_source_file = b.path("tools/fetch.zig"),
            .target = b.graph.host,
            .optimize = .Debug,
        }),
    });

    const archive = b.cache_root.join(b.allocator, &.{ "onnxruntime", openvino_archive }) catch @panic("OOM");
    const run = b.addRunArtifact(fetch_exe);
    run.has_side_effects = true;
    run.addArgs(&.{
        "--url",     openvino_url,
        "--out",     archive,
        "--sha256",  openvino_sha256,
        "--extract", openvinoRoot(b),
        "--label",   "OpenVINO " ++ openvino_version ++ " (111 MiB)",
    });

    const step = b.step("fetch-openvino", "Download Intel's OpenVINO release, for -Dopenvino-fetch");
    step.dependOn(&run.step);
    return run;
}

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

    /// `cxx` is the compiler to interrogate. It matters which: a -Dcuda build
    /// typically has to be built with a compiler older than the latest one
    /// the toolkit knows about, so -Dcuda-ccbin usually names an older one.
    fn detect(b: *std.Build, cxx: []const u8) Gnu {
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
                include.append(b.allocator, dir) catch @panic("OOM");
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

/// The dependencies, include set and generated headers that the runtime and the
/// execution providers are all built from. `b.dependency` is memoised, so
/// naming them once here is only about not repeating the list.
const Parts = struct {
    b: *std.Build,
    target: std.Build.ResolvedTarget,
    optimize: std.builtin.OptimizeMode,
    openvino: ?OpenVino,
    cuda: ?Cuda,
    cxx: []const u8,
    gnu: ?Gnu,
    ort: *std.Build.Dependency,
    protobuf: *std.Build.Dependency,
    onnx: *std.Build.Dependency,
    abseil: *std.Build.Dependency,
    re2: *std.Build.Dependency,
    cpuinfo: *std.Build.Dependency,
    protos: std.Build.LazyPath,
    includes: []const std.Build.LazyPath,
    /// Flags for every C++ translation unit in the build.
    flags: []const []const u8,

    fn init(
        b: *std.Build,
        target: std.Build.ResolvedTarget,
        optimize: std.builtin.OptimizeMode,
        openvino: ?OpenVino,
        cuda: ?Cuda,
    ) Parts {
        const ort = b.dependency("ort_src", .{});
        const protobuf = b.dependency("protobuf", .{});
        const onnx = b.dependency("onnx", .{});
        const abseil = b.dependency("abseil", .{});
        const re2 = b.dependency("re2", .{});
        const cpuinfo = b.dependency("cpuinfo", .{});

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

        const cxx = if (cuda) |c|
            c.ccbin
        else
            b.graph.environ_map.get("CXX") orelse "c++";

        // -gline-tables-only: full DWARF for a build this size overruns what
        // Mach-O's debug map can address, and the line tables are all the
        // symbolicated backtraces here need.
        const cxx_base = if (openvino == null) &ort_flags else &ort_openvino_flags;
        var cxx_flags = if (target.result.os.tag.isDarwin())
            concatFlags(b, &.{ cxx_base, &.{"-gline-tables-only"} })
        else
            cxx_base;

        if (cuda != null) {
            // The runtime and the CUDA provider meet over `ProviderHostCPU`, a pure
            // virtual interface whose *shape* depends on this define: 16 of its
            // methods sit behind `#if defined(USE_CUDA) ||
            // defined(USE_CUDA_PROVIDER_INTERFACE)` in cpu_provider_shared.h. The
            // provider is compiled with USE_CUDA and so sees the long version; the
            // runtime holds the implementation and, without this, would compile the
            // short one. Two vtable layouts over one object is not a link error --
            // every call through it simply arrives at the wrong slot, which shows up
            // as a jump to a nonsense address the first time a CUDA kernel asks the
            // CPU provider for anything. Upstream sets it in ORT_PROVIDER_FLAGS for
            // the same reason. Harmless on the provider, where the `#if` is a
            // disjunction and USE_CUDA has already satisfied it.
            cxx_flags = concatFlags(b, &.{ cxx_flags, &.{"-DUSE_CUDA_PROVIDER_INTERFACE=1"} });
        }

        return .{
            .b = b,
            .target = target,
            .optimize = optimize,
            .openvino = openvino,
            .cuda = cuda,
            .cxx = cxx,
            .gnu = if (openvino == null and cuda == null) null else Gnu.detect(b, cxx),
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

    /// A module compiled as C++ against the chosen runtime, seeded with the
    /// shared include set.
    fn cxxModule(self: Parts, target: std.Build.ResolvedTarget) *std.Build.Module {
        const mod = self.b.createModule(.{
            .target = target,
            .optimize = self.optimize,
            .link_libc = true,
            .link_libcpp = self.gnu == null,
        });
        for (self.includes) |include| mod.addIncludePath(include);
        if (self.gnu) |gnu| {
            // -I, not -isystem: Zig lists its own libc directories ahead of
            // every -isystem path, and libstdc++'s <cstdlib> reaches its
            // libc counterpart by #include_next, which only searches what
            // comes after. The C++ headers have to be found first for that
            // to land on glibc's stdlib.h.
            for (gnu.include) |dir| mod.addIncludePath(.{ .cwd_relative = dir });
        }
        return mod;
    }

    /// Add the C++ standard library to a module that is about to be linked
    /// into a shared library or an executable. `cxxModule` deliberately does
    /// not do this: most modules here end up as static archives, which is the
    /// one place these files must not go.
    fn linkCxx(self: Parts, mod: *std.Build.Module) void {
        if (self.gnu) |gnu| gnu.link(mod);
    }
    fn runtime(self: Parts) *std.Build.Step.Compile {
        const b = self.b;
        const ort_root = self.ort.path("onnxruntime");
        const target_sources = sources.forTarget(self.target.result);

        const lib_mod = self.cxxModule(self.target);
        lib_mod.addIncludePath(self.protos);

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

        // Travels with the artifact: a module that links this library gets the C
        // API headers on its include path without naming a path itself.
        lib.installHeadersDirectory(self.ort.path("include/onnxruntime/core/session"), "", .{});
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
            .root = self.ort.path("onnxruntime"),
            .files = &sources.ort_provider_host_sources,
            .flags = self.flags,
        });
        self.linkCxx(mod);
        const shared = b.addLibrary(.{
            .name = "onnxruntime_providers_shared",
            .linkage = .dynamic,
            .root_module = mod,
        });
        shared.setVersionScript(self.ort.path("onnxruntime/core/providers/shared/version_script.lds"));
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
        const ort_root = self.ort.path("onnxruntime");

        const mod = self.cxxModule(self.target);
        mod.addIncludePath(self.protos);
        mod.addIncludePath(.{ .cwd_relative = openvino.include });

        const flags_ = concatFlags(b, &.{
            self.flags,
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
        mod.addCSourceFiles(.{ .root = ort_root, .files = &sources.ort_openvino_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = ort_root, .files = &sources.ort_provider_shared_sources, .flags = flags_ });

        // The provider carries its own onnx protobuf and abseil rather than
        // sharing the runtime's: the version script below hides every symbol
        // that is not an entry point, so the two copies cannot collide.
        mod.addCSourceFiles(.{ .root = self.protos, .files = &sources.onnx_proto_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = self.protobuf.path(""), .files = &sources.protobuf_lite_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = self.abseil.path(""), .files = &sources.abseil_sources, .flags = flags_ });

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
        provider.setVersionScript(self.ort.path("onnxruntime/core/providers/openvino/version_script.lds"));
        provider.root_module.linkLibrary(shared);
        // The runtime dlopens providers_shared with RTLD_GLOBAL before any
        // provider, so by this point the symbol is already in the process.
        // $ORIGIN covers the case where it is not: the two are installed
        // side by side.
        provider.root_module.addRPath(.{ .cwd_relative = "$ORIGIN" });
        // A downloaded OpenVINO is not on disk until its step has run, and the
        // headers above are inside it -- so the compile waits on the download
        // rather than the install.
        if (openvino.fetch) |fetch| provider.step.dependOn(&fetch.step);
        return provider;
    }

    /// libonnxruntime_providers_cuda.so.
    ///
    /// Same shape as `openvinoProvider` -- a dlopened .so exporting
    /// GetProvider, CreateEpFactories and ReleaseEpFactory, reaching the
    /// runtime through `providersShared` -- but built from two halves rather
    /// than one. The `.cc` files are ordinary C++ and go through Zig like
    /// everything else here; the `.cu` files hold device code, which only nvcc
    /// can emit, so each is compiled to an object by `deviceObjects` and handed
    /// to the linker.
    ///
    /// Scope: both operator trees, ONNX and com.microsoft. See
    /// `sources.ort_cuda_sources` for why the second is not optional.
    fn cudaProvider(self: Parts, shared: *std.Build.Step.Compile) *std.Build.Step.Compile {
        const b = self.b;
        const cuda = self.cuda.?;
        const ort_root = self.ort.path("onnxruntime");

        const mod = self.cxxModule(self.target);
        mod.addIncludePath(self.protos);
        for (self.cudaIncludes()) |include| mod.addIncludePath(include);

        const flags_ = concatFlags(b, &.{ self.flags, &cuda_defines, self.cudaConfigDefines() });
        mod.addCSourceFiles(.{ .root = ort_root, .files = &sources.ort_cuda_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = ort_root, .files = &sources.ort_provider_shared_sources, .flags = flags_ });

        // Its own onnx protobuf and abseil, for the reason `openvinoProvider`
        // gives: the version script hides everything that is not an entry
        // point, so the runtime's copies and these cannot collide.
        mod.addCSourceFiles(.{ .root = self.protos, .files = &sources.onnx_proto_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = self.protobuf.path(""), .files = &sources.protobuf_lite_sources, .flags = flags_ });
        mod.addCSourceFiles(.{ .root = self.abseil.path(""), .files = &sources.abseil_sources, .flags = flags_ });

        for (self.deviceObjects()) |object| mod.addObjectFile(object);

        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda.prefix, "lib" }) });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda.prefix, "lib64" }) });
        // cuDNN and cuFFT are deliberately absent: the provider dlopens them by
        // soname when an op needs them (see cudnn_loader.cc), so a machine that
        // never runs a convolution needs neither installed.
        for ([_][]const u8{ "cudart", "cublas", "cublasLt", "curand" }) |name| {
            mod.linkSystemLibrary(name, .{});
        }
        // libcuda.so.1 is the driver, not the toolkit. It ships with the
        // installed NVIDIA driver, in a directory that has nothing to do with
        // this prefix and differs on every distribution, so the toolkit carries
        // a stub -- the right symbols over no implementation -- to link against
        // instead. At run time the loader finds the real one, which is why the
        // stub directory is deliberately not an RPATH below: resolving there at
        // run time would find a driver that can do nothing.
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda.prefix, "lib", "stubs" }) });
        mod.addLibraryPath(.{ .cwd_relative = b.pathJoin(&.{ cuda.prefix, "lib64", "stubs" }) });
        mod.linkSystemLibrary("cuda", .{});
        mod.addRPath(.{ .cwd_relative = b.pathJoin(&.{ cuda.prefix, "lib" }) });
        self.linkCxx(mod);

        const provider = b.addLibrary(.{
            .name = "onnxruntime_providers_cuda",
            .linkage = .dynamic,
            .root_module = mod,
        });
        provider.setVersionScript(self.ort.path("onnxruntime/core/providers/cuda/version_script.lds"));
        provider.root_module.linkLibrary(shared);
        provider.root_module.addRPath(.{ .cwd_relative = "$ORIGIN" });
        return provider;
    }

    /// The include directories the CUDA provider needs on top of `includes`:
    /// the toolkit's own headers, cuDNN's, and the two header-only NVIDIA
    /// libraries the provider's convolution and attention kernels are written
    /// against. Both of the latter are lazy dependencies -- they are 200 MB
    /// between them and no other build here touches either.
    fn cudaIncludes(self: Parts) []const std.Build.LazyPath {
        const b = self.b;
        const cuda = self.cuda.?;

        var list: std.ArrayList(std.Build.LazyPath) = .empty;
        list.appendSlice(b.allocator, &.{
            .{ .cwd_relative = b.pathJoin(&.{ cuda.prefix, "include" }) },
            .{ .cwd_relative = cuda.cudnn_include },
        }) catch @panic("OOM");
        if (b.lazyDependency("cudnn_frontend", .{})) |dep| {
            list.append(b.allocator, dep.path("include")) catch @panic("OOM");
        }
        if (b.lazyDependency("cutlass", .{})) |dep| {
            list.appendSlice(b.allocator, &.{
                dep.path("include"),
                // Not a mistake: the fused multi-head attention kernels include
                // headers out of CUTLASS's example 41, which upstream also puts
                // on the include path rather than vendoring.
                dep.path("examples"),
                dep.path("tools/util/include"),
            }) catch @panic("OOM");
        }
        return list.items;
    }

    /// The defines that depend on the build, as opposed to the fixed
    /// `cuda_defines`: what the optimiser was told, and what the target
    /// architecture can do.
    fn cudaConfigDefines(self: Parts) []const []const u8 {
        const b = self.b;

        var list: std.ArrayList([]const u8) = .empty;
        // NDEBUG is not a nicety here. xqa's kernel_mha_impl is a plain
        // __device__ function with it and a __global__ kernel without, and the
        // second form makes nvcc emit a host stub naming a type it cannot spell.
        // Zig defines this for its own release compiles; nvcc has to be told.
        if (self.optimize != .Debug) list.appendSlice(b.allocator, &.{ "-O3", "-DNDEBUG" }) catch @panic("OOM");

        // Which SM the kernels may assume. During nvcc's *device* pass the code
        // reads __CUDA_ARCH__ directly, but on the host pass there is no such
        // macro, and xqa keys its kernel declarations off these instead. Get it
        // wrong and the two passes disagree about which kernels exist -- the
        // device side defines one the host side never declares, and the
        // generated stub refers to a member of a namespace that has none.
        const sm = std.fmt.parseInt(u32, self.cuda.?.arch, 10) catch {
            std.log.err("-Dcuda-arch must be a compute capability such as 89, not '{s}'", .{self.cuda.?.arch});
            std.process.exit(1);
        };
        if (sm >= 80) list.append(b.allocator, "-DHAS_SM80_OR_LATER") catch @panic("OOM");
        if (sm >= 90) list.append(b.allocator, "-DHAS_SM90_OR_LATER") catch @panic("OOM");

        // Which narrow float types the provider is compiled to know about.
        // These describe the *types*, not the quantised-MoE kernels that use
        // them -- those are a separate pair of options, left off, and the
        // kernels behind them are not in the source lists. Both matter even so:
        // moe_gemm_kernels.h declares use_wfp4afp8 in the fallback arm of the
        // FP8 block and again in the fallback arm of the FP4 block, so with
        // neither type enabled it declares the same member twice.
        list.appendSlice(b.allocator, &.{ "-DENABLE_BF16", "-DENABLE_FP8" }) catch @panic("OOM");
        if (self.cuda.?.version(b) >= 1208) list.append(b.allocator, "-DENABLE_FP4") catch @panic("OOM");
        return list.items;
    }

    /// One object per `.cu` file, each from its own nvcc run.
    ///
    /// A run step per translation unit rather than one nvcc over the lot: it is
    /// what lets the build system schedule them across cores and skip the ones
    /// whose inputs have not moved. nvcc is slow enough -- ten seconds and up
    /// for a kernel of any size -- that both matter.
    fn deviceObjects(self: Parts) []const std.Build.LazyPath {
        const b = self.b;
        const cuda = self.cuda.?;

        const arch = b.fmt("-gencode=arch=compute_{s},code=sm_{s}", .{ cuda.arch, cuda.arch });
        var objects: std.ArrayList(std.Build.LazyPath) = .empty;
        for (sources.ort_cuda_device_sources) |file| {
            const run = b.addSystemCommand(&.{ cuda.compiler(b), "-ccbin", cuda.ccbin });
            run.addArgs(&.{ "-std=c++20", "-c", arch });
            run.addArgs(&nvcc_flags);
            run.addArgs(&cuda_defines);
            run.addArgs(self.cudaConfigDefines());
            // The .cu objects land in a shared library beside the .cc ones, so
            // the host half of each has to be built the same way.
            run.addArgs(&.{ "-Xcompiler", "-fPIC" });

            for (self.includes) |include| run.addPrefixedDirectoryArg("-I", include);
            run.addPrefixedDirectoryArg("-I", self.protos);
            for (self.cudaIncludes()) |include| run.addPrefixedDirectoryArg("-I", include);

            run.addFileArg(self.ort.path(b.fmt("onnxruntime/{s}", .{file})));
            run.addArg("-o");
            objects.append(
                b.allocator,
                run.addOutputFileArg(b.fmt("{s}.o", .{std.fs.path.basename(file)})),
            ) catch @panic("OOM");
        }
        return objects.items;
    }
};

/// Defines shared by both halves of the CUDA provider.
const cuda_defines = [_][]const u8{
    "-DUSE_CUDA=1",
    // Both on, as upstream has them by default for a CUDA build. They gate the
    // two attention backends the ONNX-domain Attention kernel is written
    // against, and the sources behind them are in `ort_cuda_device_sources`.
    "-DUSE_FLASH_ATTENTION=1",
    "-DUSE_MEMORY_EFFICIENT_ATTENTION=1",
    // cudnn_frontend resolves cuDNN's entry points with dlsym rather than
    // linking them, which is what lets the provider load on a machine that has
    // no cuDNN until something asks for a convolution.
    "-DNV_CUDNN_FRONTEND_USE_DYNAMIC_LOADING",
};

/// nvcc's own switches, as opposed to the preprocessor defines above.
const nvcc_flags = [_][]const u8{
    // ORT calls plenty of constexpr host functions from device code and relies
    // on nvcc allowing it.
    "--expt-relaxed-constexpr",
    "--expt-extended-lambda",
    // CUDA 12.8 turned several long-standing patterns in this tree into
    // diagnostics: 177 unused variable in CUTLASS, 221 and 550 from the CUDA
    // and flatbuffers headers, 2810 a false positive on assigning a
    // [[nodiscard]] Status, 2908 deprecated by-copy `this` capture in CUTLASS.
    // 68, 69 and 554 are abseil's and GSL's, and predate that.
    "--diag-suppress=68,69,177,221,554,2810,2908",
    "-Xcudafe",
    "--diag_suppress=550",
    // Since 12.8 nvcc emits a definition for every global template stub it
    // sees, which multiply-defines the ones ORT instantiates in more than one
    // translation unit.
    "--static-global-template-stub=false",
    "-Xcompiler",
    "-Wno-reorder",
};

fn buildProtoc(b: *std.Build, protobuf: *std.Build.Dependency) *std.Build.Step.Compile {
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
    protoc_mod.addIncludePath(protobuf.path("src"));
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
        protoc_mod.addCSourceFiles(.{ .root = protobuf.path(""), .files = list, .flags = &flags });
    }
    return b.addExecutable(.{ .name = "protoc", .root_module = protoc_mod });
}

fn generateOnnxProto(b: *std.Build, protoc_exe: *std.Build.Step.Compile, onnx: *std.Build.Dependency) std.Build.LazyPath {
    const run = b.addRunArtifact(protoc_exe);
    for ([_][]const u8{ "onnx-ml.proto", "onnx-operators-ml.proto", "onnx-data.proto" }) |proto| {
        run.addFileArg(onnx.path(b.fmt("onnx/{s}", .{proto})));
    }
    run.addArg("-I");
    run.addDirectoryArg(onnx.path(""));
    run.addArg("--cpp_out");
    return run.addOutputDirectoryArg("onnx-proto");
}

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

// OpenVINO's headers throw and catch by type, and its own build has RTTI on,
// so the no-RTTI pair is omitted.
//
// -Wno-invalid-constexpr: glibc 2.41 dropped __CORRECT_ISO_CPP_MATH_H_PROTO,
// so libstdc++'s <cmath> now declares its own constexpr acos(float) and friends
// over __builtin_acosf. GCC folds those at compile time; clang does not, and
// rejects the definitions outright rather than warning.
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
