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

```
$ ldd your-program
    linux-vdso.so.1
    libm.so.6 => ...
    libc.so.6 => ...
    ld-linux-x86-64.so.2
```

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

ONNX Runtime 1.24.4, CPU execution provider, static. MLAS is compiled once per
instruction-set group — each group a module with its own `cpu_features_add` —
so the runtime still dispatches on AVX2, AVX-512 and the rest at run time
while the rest of the library stays at the target's baseline.

The first build compiles a few thousand C++ translation units and takes a
while. Later builds come out of the Zig cache.

## Version

The runtime version is `.version` in `build.zig.zon`. Moving to a new ONNX
Runtime means bumping `ort_src` and reconciling `sources.zig` with upstream's
CMake, which does drift between releases.
