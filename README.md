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

Enable the Intel NPU/GPU provider with `zig build -Dopenvino`. The build
downloads its pinned OpenVINO release automatically.

Consumers must call `linkStdCxx`, install `onnxruntime_providers_shared` beside
the executable, and install `OpenCL` when using the GPU. Use
`addOpenVinoRuntimeEnvironment` to configure the run step:

```zig
onnxruntime.linkStdCxx(b, exe.root_module);
onnxruntime.addOpenVinoRuntimeEnvironment(b, run, .gpu, extra, known, null);
```
