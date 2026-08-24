# onnxruntime

ONNX Runtime built from source by the Zig build system, as a Zig package.

```
build.zig      the build: protoc, the runtime, MLAS ISA groups, flags
sources.zig    every source file compiled, transcribed from upstream CMake
```

No CMake, no Python, no configure step, and nothing installed on the machine:
protobuf, onnx, abseil, re2, cpuinfo and the runtime itself are compiled from
pinned source archives. The C++ runtime comes from Zig's own libc++, so a
binary linking this depends on no system library beyond libc.

The exception is [Intel NPU support](#intel-npu), which is opt-in and links a
system OpenVINO, because the NPU is only reachable through a compiler Intel
ships as a binary.

## Using it

`build.zig.zon`:

```zig
.dependencies = .{
    .onnxruntime = .{
        .url = "git+https://github.com/kirillrdy/onnxruntime#<commit>",
        .hash = "...",
    },
},
```

`build.zig`:

```zig
const ort = b.dependency("onnxruntime", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.linkLibrary(ort.artifact("onnxruntime"));
```

Linking the artifact carries its headers along, so the C API needs no include
path of your own:

```zig
const c = @cImport({
    @cInclude("onnxruntime_c_api.h");
});
```

## What is built

The CPU execution provider, static, for x86-64 POSIX targets. MLAS is compiled
once per instruction-set group — each group a module with its own
`cpu_features_add` — so the runtime still dispatches on AVX2, AVX-512 and the
rest at run time while the rest of the library stays at the target's baseline.

The first build compiles a few thousand C++ translation units and takes a
while. Later builds come out of the Zig cache.

## Intel NPU

`-Dopenvino=<prefix>` additionally builds the OpenVINO execution provider,
which is how ONNX Runtime reaches an Intel NPU:

```
zig build -Dopenvino=/path/to/openvino
```

It produces three artifacts instead of one, and a consumer wires them up like
this:

```zig
const onnxruntime = @import("onnxruntime");

const ort = b.dependency("onnxruntime", .{
    .target = target,
    .optimize = optimize,
    .openvino = @as([]const u8, "/path/to/openvino"),
});
exe.root_module.linkLibrary(ort.artifact("onnxruntime"));

// This build runs on the host's libstdc++ rather than Zig's libc++, and Zig
// does not carry a static library's link objects across to whoever links it.
onnxruntime.linkStdCxx(b, exe.root_module);

// The runtime looks for this beside the executable, not in lib/.
b.getInstallStep().dependOn(&b.addInstallArtifact(
    ort.artifact("onnxruntime_providers_shared"),
    .{ .dest_dir = .{ .override = .bin } },
).step);
```

`libonnxruntime_providers_shared.so` is one global pointer and nothing else.
The runtime dlopens it by name, out of the directory the running binary sits
in, and leaves a vtable of its internals there; each provider links the same
file and picks the vtable back up. Miss it and loading a provider fails with
`undefined symbol: Provider_GetHost`; miss `linkStdCxx` and the link fails
with undefined `__cxa_begin_catch`.

The third artifact, `libonnxruntime_providers_openvino.so`, the runtime does
not link at all — it dlopens it by path when asked. So a program registers it
at run time and then picks the NPU out of the enumerated devices:

```c
g->RegisterExecutionProviderLibrary(env, "OpenVINO", so_path);
g->GetEpDevices(env, &devices, &num_devices);
/* find one whose HardwareDevice_Type is OrtHardwareDeviceType_NPU */
g->SessionOptionsAppendExecutionProvider_V2(opts, env, &npu, 1, NULL, NULL, 0);
```

That enumerates the iGPU too, so the same flag gets you `GPU` as well as `NPU`.

`-Dopenvino-include=<dir>` and `-Dopenvino-lib=<dir>` point at the headers and
at `libopenvino.so` separately, overriding `<prefix>/include` and
`<prefix>/lib`. The first is for package managers that split a build into
`dev` and `lib` outputs; the second is for Intel's own archives, which put the
libraries and the plugins together in `runtime/lib/intel64` — a layout no
prefix describes.

### What it costs

This build gives up the properties the rest of the README describes, because
an Intel NPU can only be reached through Intel's own compiler, which ships as
a prebuilt binary inside the OpenVINO install. There is no from-source path.

Since the provider and the runtime hand each other C++ objects across the
dlopen boundary, both have to match the ABI of the prebuilt `libopenvino.so`
— GCC's libstdc++, with RTTI on. So `-Dopenvino` switches the *whole* build
off Zig's libc++ and onto the host's libstdc++, found by asking `c++` (or
`$CXX`) where its headers and libraries are. That build links libstdc++,
libgcc_s and libopenvino, and cannot cross-compile; without the flag nothing
changes.

### Requirements

OpenVINO 2026.0 or newer — the version ONNX Runtime 1.29 requires — with the
NPU plugin and Intel's NPU compiler (`libopenvino_intel_npu_compiler.so`)
present alongside the plugins. Some distributions package the plugin without
the compiler, in which case compiling for the NPU fails with `VCL compiler
loading failed`; Intel's own archives carry both.

At run time the NPU plugin needs the Level Zero loader (`libze_loader.so.1`)
and the NPU driver (`libze_intel_npu.so.1`) on the library path. If only the
loader is reachable the NPU still enumerates but every compile fails with `No
available backend`.

## NVIDIA CUDA (GPU)

`-Ddevice=cuda` (or `-Dcuda-fetch`) builds the CUDA execution provider from source, downloading NVIDIA's CUDA 13.0 toolkit and cuDNN 9.14 redistributables into the build cache:

```
zig build -Ddevice=cuda
```

Alternatively, if you have a local CUDA installation:

```
zig build -Dcuda=/path/to/cuda
```

Consumers can select CUDA effortlessly using `select()`:

```zig
const onnxruntime = @import("onnxruntime");

// Declares -Ddevice, -Dcuda, etc. Automatically downloads toolkit if needed:
const ort = onnxruntime.select(b, target, optimize, .cuda);

const exe = b.addExecutable(.{ ... });
ort.addImport(exe.root_module);
ort.linkExecutable(exe);
_ = ort.installProvider();
```

### Options

| Option | Default | Description |
|---|---|---|
| `-Ddevice=cuda` | `cpu` | Target execution provider |
| `-Dcuda-fetch` | `false` | Explicitly download NVIDIA CUDA 13.0 & cuDNN 9.14 redistributables |
| `-Dcuda=<prefix>` | `null` | Path to an existing local CUDA installation |
| `-Dcuda-arch=<sm>` | `89` (Ada) | Target compute capability (`75`, `80`, `86`, `89`, `90`) |
| `-Dcuda-ccbin=<c++>` | first supported g++ | Host C++ compiler for nvcc |

The toolkit only drives a GCC its `crt/host_config.h` allows -- 15 and older for
CUDA 13.0. `-Dcuda-ccbin` is otherwise worked out from `$CXX`, `c++`, `g++` and
the versioned `g++-N` names, and the build stops at configuration if none of
them fits, rather than deep inside libstdc++ once nvcc is already running.

## Version

`.version` in `build.zig.zon` is the ONNX Runtime version. Moving to a new
release means bumping `ort_src` and reconciling `sources.zig` with upstream's
CMake, which does drift between releases.
