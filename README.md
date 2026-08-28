# onnxruntime

ONNX Runtime built from source as a Zig package. The default static library
uses Zig's libc++ and supports x86-64 POSIX and AArch64 Linux targets.

Add the dependency to `build.zig.zon`:

```zig
.dependencies = .{
    .onnxruntime = .{
        .url = "git+https://github.com/kirillrdy/onnxruntime#<commit>",
        .hash = "...",
    },
},
```

Link it in `build.zig`:

```zig
const ort = b.dependency("onnxruntime", .{
    .target = target,
    .optimize = optimize,
});
exe.root_module.linkLibrary(ort.artifact("onnxruntime"));
```

## OpenVINO

Enable the Intel GPU provider with `zig build -Dopenvino=true`. OpenVINO, its
GPU plugin, and the OpenCL ICD loader are built from pinned source dependencies.
The source build currently supports native x86-64 Linux targets.

The GPU plugin reads its OpenCL kernels out of generated `.inc` databases.
`tools/cl_kernel_db.zig` builds them during the build, from the pinned OpenVINO
sources; it is a port of the two Python scripts OpenVINO's CMake build uses for
the same job, so the build still needs nothing but Zig.

Consumers must install `onnxruntime_providers_shared` and
`onnxruntime_providers_openvino` beside the executable. Use
`addOpenVinoRuntimeEnvironment` to expose the built libraries and configure an
optional Intel OpenCL driver for the run step:

```zig
onnxruntime.addOpenVinoRuntimeEnvironment(b, run, .gpu, extra, known, null);
```
