//! Downloader behind `zig build fetch-openvino`, which pulls Intel's OpenVINO
//! release into the build cache.
//!
//! It exists because that archive cannot be a `build.zig.zon` dependency: see
//! `download` for what Zig's own HTTP client and `zig fetch` both run into
//! against Intel's CDN. Everything else this package builds is a zon
//! dependency, and would be here too if it could be.
//!
//! Usage:
//!   fetch --url <url> --out <path> [--sha256 <hex>] [--label <text>] [--extract <dir>]
//!
//! With `--sha256` the download is verified and the file is only moved into
//! place once it matches; an existing file that already matches is left alone.
//! `--extract` unpacks it, and a destination that is already there is left
//! alone the same way -- so the step is idempotent and cheap to re-run.

const std = @import("std");

const Options = struct {
    url: []const u8,
    out: []const u8,
    sha256: ?[]const u8 = null,
    label: ?[]const u8 = null,
    /// Where to unpack `out`, for a download that is an archive rather than a
    /// file to be used as it is. The top-level directory the archive carries
    /// its own version in is stripped, so this directory *is* the archive.
    extract: ?[]const u8 = null,
};

pub fn main(init: std.process.Init) !void {
    const gpa = init.gpa;
    const io = init.io;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    defer args_it.deinit();
    _ = args_it.skip();

    var url: ?[]const u8 = null;
    var out: ?[]const u8 = null;
    var sha256: ?[]const u8 = null;
    var label: ?[]const u8 = null;
    var extract: ?[]const u8 = null;

    while (args_it.next()) |arg| {
        if (std.mem.eql(u8, arg, "--url")) {
            url = args_it.next();
        } else if (std.mem.eql(u8, arg, "--out")) {
            out = args_it.next();
        } else if (std.mem.eql(u8, arg, "--sha256")) {
            sha256 = args_it.next();
        } else if (std.mem.eql(u8, arg, "--label")) {
            label = args_it.next();
        } else if (std.mem.eql(u8, arg, "--extract")) {
            extract = args_it.next();
        } else {
            std.debug.print("fetch: unknown argument '{s}'\n", .{arg});
            return error.InvalidArguments;
        }
    }

    const opts = Options{
        .url = url orelse return error.InvalidArguments,
        .out = out orelse return error.InvalidArguments,
        .sha256 = sha256,
        .label = label,
        .extract = extract,
    };

    const name = opts.label orelse opts.out;

    if (opts.sha256) |want| {
        if (try hashFile(io, opts.out)) |have| {
            if (std.ascii.eqlIgnoreCase(&have, want)) {
                return unpack(gpa, io, opts, name);
            }
            std.debug.print("  {s}: present but checksum differs, re-downloading\n", .{name});
        }
    } else if (fileExists(io, opts.out)) {
        return unpack(gpa, io, opts, name);
    }

    if (std.fs.path.dirname(opts.out)) |dir| {
        try std.Io.Dir.cwd().createDirPath(io, dir);
    }

    var part_buf: [4096]u8 = undefined;
    const part_path = try std.fmt.bufPrint(&part_buf, "{s}.part", .{opts.out});

    std.debug.print("  {s}: downloading\n", .{name});
    try download(gpa, io, opts, part_path);

    if (opts.sha256) |want| {
        const have = (try hashFile(io, part_path)) orelse return error.DownloadDisappeared;
        if (!std.ascii.eqlIgnoreCase(&have, want)) {
            std.debug.print(
                \\  {s}: SHA-256 mismatch
                \\    expected {s}
                \\    actual   {s}
                \\
            , .{ name, want, &have });
            std.Io.Dir.cwd().deleteFile(io, part_path) catch {};
            return error.ChecksumMismatch;
        }
        std.debug.print("  {s}: verified against the published SHA-256\n", .{name});
    }

    const cwd = std.Io.Dir.cwd();
    try cwd.rename(part_path, cwd, opts.out, io);
    std.debug.print("  {s}: -> {s}\n", .{ name, opts.out });

    try unpack(gpa, io, opts, name);
}

/// Unpack `opts.out` into `opts.extract`, if this download is an archive.
///
/// Idempotent the same way the download is: the destination existing is what
/// says the work was done, so re-running the step costs a stat. The archive is
/// unpacked beside it and renamed into place, so an interrupted run leaves no
/// half-populated directory to be mistaken for a finished one.
fn unpack(gpa: std.mem.Allocator, io: std.Io, opts: Options, name: []const u8) !void {
    const dest = opts.extract orelse return;
    const cwd = std.Io.Dir.cwd();

    if (cwd.openDir(io, dest, .{})) |existing| {
        var dir = existing;
        dir.close(io);
        return;
    } else |_| {}

    var part_buf: [4096]u8 = undefined;
    const part = try std.fmt.bufPrint(&part_buf, "{s}.part", .{dest});
    cwd.deleteTree(io, part) catch {};
    try cwd.createDirPath(io, part);

    std.debug.print("  {s}: unpacking\n", .{name});

    var archive = try cwd.openFile(io, opts.out, .{});
    defer archive.close(io);

    const read_buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(read_buf);
    var archive_reader = archive.reader(io, read_buf);

    // Zig's inflate wants the whole 64 KiB window up front; without it the
    // decompressor has nowhere to resolve back-references from.
    const window = try gpa.alloc(u8, std.compress.flate.max_window_len);
    defer gpa.free(window);
    var gzip: std.compress.flate.Decompress = .init(&archive_reader.interface, .gzip, window);

    var dir = try cwd.openDir(io, part, .{});
    defer dir.close(io);

    // One component stripped: the archive carries its own build number as a
    // top-level directory, and nothing downstream should have to know it.
    try std.tar.extract(io, dir, &gzip.reader, .{ .strip_components = 1 });

    try cwd.rename(part, cwd, dest, io);
    std.debug.print("  {s}: -> {s}/\n", .{ name, dest });
}

fn fileExists(io: std.Io, path: []const u8) bool {
    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, path, .{}) catch return false;
    file.close(io);
    return true;
}

/// curl first, Zig's own client second.
///
/// Not the order the file's name suggests, and deliberate: `std.http.Client`
/// cannot reach the host this fetches from. Its TLS fails the handshake against
/// the post-quantum key exchange (X25519MLKEM768) that Intel's CDN negotiates,
/// which is the same reason the archive cannot be a `build.zig.zon` dependency
/// -- `zig fetch` fails there too. The fallback is kept for the day that lands.
fn download(gpa: std.mem.Allocator, io: std.Io, opts: Options, dest_path: []const u8) !void {
    downloadWithCurl(gpa, io, opts, dest_path) catch {
        return downloadZig(gpa, io, opts, dest_path);
    };
}

fn downloadWithCurl(gpa: std.mem.Allocator, io: std.Io, opts: Options, dest_path: []const u8) !void {
    const result = try std.process.run(gpa, io, .{
        .argv = &.{ "curl", "-sSL", "--retry", "3", "-o", dest_path, opts.url },
    });
    defer {
        gpa.free(result.stdout);
        gpa.free(result.stderr);
    }
    if (result.term != .exited or result.term.exited != 0) return error.HttpRequestFailed;
}

fn downloadZig(gpa: std.mem.Allocator, io: std.Io, opts: Options, dest_path: []const u8) !void {
    const cwd = std.Io.Dir.cwd();

    var file = try cwd.createFile(io, dest_path, .{});
    defer file.close(io);

    const write_buf = try gpa.alloc(u8, 1 << 20);
    defer gpa.free(write_buf);
    var file_writer = file.writer(io, write_buf);

    var client: std.http.Client = .{ .allocator = gpa, .io = io };
    defer client.deinit();

    const result = try client.fetch(.{
        .location = .{ .url = opts.url },
        .method = .GET,
        .response_writer = &file_writer.interface,
    });

    try file_writer.interface.flush();

    if (result.status != .ok) {
        std.debug.print("  HTTP {d} for {s}\n", .{ @intFromEnum(result.status), opts.url });
        return error.HttpRequestFailed;
    }
}

/// SHA-256 of a file as lowercase hex, or null when the file does not exist.
fn hashFile(io: std.Io, path: []const u8) !?[64]u8 {
    const cwd = std.Io.Dir.cwd();
    var file = cwd.openFile(io, path, .{}) catch |err| switch (err) {
        error.FileNotFound => return null,
        else => return err,
    };
    defer file.close(io);

    var read_buf: [65536]u8 = undefined;
    var file_reader = file.reader(io, &read_buf);

    var hasher = std.crypto.hash.sha2.Sha256.init(.{});
    var chunk: [65536]u8 = undefined;
    while (true) {
        const n = file_reader.interface.readSliceShort(&chunk) catch |err| switch (err) {
            error.ReadFailed => return error.ReadFailed,
        };
        if (n == 0) break;
        hasher.update(chunk[0..n]);
    }

    var digest: [32]u8 = undefined;
    hasher.final(&digest);

    var hex: [64]u8 = undefined;
    _ = std.fmt.bufPrint(&hex, "{x}", .{&digest}) catch unreachable;
    return hex;
}
