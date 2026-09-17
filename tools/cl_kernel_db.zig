//! Generates the Intel GPU plugin's OpenCL kernel databases.
//!
//! OpenVINO's CMake build runs two Python scripts to stringify the `.cl` kernels
//! into `.inc` files that `primitive_db.cpp` and `ocl_v2/utils/kernels_db.cpp`
//! `#include`:
//!
//!   src/plugins/intel_gpu/src/kernel_selector/primitive_db_gen.py
//!       -> ks_primitive_db.inc, ks_primitive_db_batch_headers.inc
//!   src/plugins/intel_gpu/src/graph/common_utils/kernels_db_gen.py
//!       -> gpu_ocl_kernel_sources.inc, gpu_ocl_kernel_headers.inc
//!
//! This is a port of both, so the build needs no Python interpreter and the
//! databases always match the pinned OpenVINO sources instead of a checked-in
//! snapshot. The Python originals lean on regular expressions; the equivalents
//! here are hand-rolled scanners, and each one names the pattern it replaces.
//!
//! Each mode writes its `.inc` files into `<out_dir>` under the names the
//! including translation units expect.
//!
//! Usage:
//!   cl_kernel_db primitive-db <cl_kernels_dir> <out_dir>
//!   cl_kernel_db ocl-v2 <kernels_dir> <headers_dir> <out_dir>

const std = @import("std");
const Allocator = std.mem.Allocator;
const Dir = std.Io.Dir;

const Buf = std.ArrayList(u8);
const List = std.ArrayList([]const u8);
const Set = std.StringArrayHashMapUnmanaged(void);

const max_file_bytes = std.Io.Limit.limited(16 << 20);

pub fn main(init: std.process.Init) !void {
    var arena_state: std.heap.ArenaAllocator = .init(init.gpa);
    defer arena_state.deinit();
    const gpa = arena_state.allocator();
    const io = init.io;

    var args_it = try std.process.Args.Iterator.initAllocator(init.minimal.args, gpa);
    _ = args_it.skip();

    const mode = args_it.next() orelse return usage();
    if (std.mem.eql(u8, mode, "primitive-db")) {
        const kernels = args_it.next() orelse return usage();
        const out_dir = args_it.next() orelse return usage();
        try generatePrimitiveDb(gpa, io, kernels, out_dir);
    } else if (std.mem.eql(u8, mode, "ocl-v2")) {
        const kernels = args_it.next() orelse return usage();
        const headers = args_it.next() orelse return usage();
        const out_dir = args_it.next() orelse return usage();
        try generateOclV2(gpa, io, kernels, headers, out_dir);
    } else {
        return usage();
    }
}

fn usage() error{InvalidArguments} {
    std.debug.print(
        \\usage: cl_kernel_db primitive-db <cl_kernels_dir> <out_dir>
        \\       cl_kernel_db ocl-v2 <kernels_dir> <headers_dir> <out_dir>
        \\
    , .{});
    return error.InvalidArguments;
}

fn writeOutput(io: std.Io, out_dir: []const u8, name: []const u8, data: []const u8) !void {
    var dir = try Dir.cwd().openDir(io, out_dir, .{});
    defer dir.close(io);
    try dir.writeFile(io, .{ .sub_path = name, .data = data });
}

// ---------------------------------------------------------------------------
// Python string and regex primitives
// ---------------------------------------------------------------------------

/// The character class Python's `\s` matches for byte strings; identical to
/// `std.ascii.whitespace`.
const isSpace = std.ascii.isWhitespace;

/// The character class Python's `\w` matches for byte strings.
fn isWord(c: u8) bool {
    return std.ascii.isAlphanumeric(c) or c == '_';
}

fn strip(s: []const u8) []const u8 {
    return std.mem.trim(u8, s, &std.ascii.whitespace);
}

fn rstrip(s: []const u8) []const u8 {
    return std.mem.trimEnd(u8, s, &std.ascii.whitespace);
}

/// End of the `\s*` run starting at `i`.
fn skipSpace(s: []const u8, i: usize) usize {
    return std.mem.findNonePos(u8, s, i, &std.ascii.whitespace) orelse s.len;
}

/// End of the `\s+` run starting at `i`, or null when there is none.
fn skipSpace1(s: []const u8, i: usize) ?usize {
    const j = skipSpace(s, i);
    return if (j == i) null else j;
}

/// End of the `[^\S\n]*` run starting at `i`: whitespace short of a line break.
fn skipHSpace(s: []const u8, i: usize) usize {
    return std.mem.findNonePos(u8, s, i, " \t\r\x0b\x0c") orelse s.len;
}

/// End of the `\w*` run starting at `i`.
fn skipWord(s: []const u8, i: usize) usize {
    var j = i;
    while (j < s.len and isWord(s[j])) j += 1;
    return j;
}

/// `\s*\n` anchored at `i`, as the regex engine resolves it: the greedy run is
/// given back until it ends on a line break, so the match covers the run up to
/// and including its *last* newline. Returns null when the run holds none.
fn spaceRunThroughNewline(s: []const u8, i: usize) ?usize {
    const end = skipSpace(s, i);
    var k = end;
    while (k > i) {
        k -= 1;
        if (s[k] == '\n') return k + 1;
    }
    return null;
}

fn startsWithAt(s: []const u8, i: usize, needle: []const u8) bool {
    return i + needle.len <= s.len and std.mem.eql(u8, s[i..][0..needle.len], needle);
}

/// Python `str.split("\n")`: keeps the trailing empty element.
fn splitRaw(gpa: Allocator, s: []const u8) ![][]const u8 {
    var out: List = .empty;
    var it = std.mem.splitScalar(u8, s, '\n');
    while (it.next()) |line| try out.append(gpa, line);
    return out.toOwnedSlice(gpa);
}

/// Python `str.splitlines()`: no trailing empty element for a trailing newline.
fn splitLines(gpa: Allocator, s: []const u8) ![][]const u8 {
    const lines = try splitRaw(gpa, s);
    return if (lines[lines.len - 1].len == 0) lines[0 .. lines.len - 1] else lines;
}

fn joinLines(gpa: Allocator, parts: []const []const u8) ![]const u8 {
    return std.mem.join(gpa, "\n", parts);
}

/// Drops empty lines, per Python's `if line` filter, or with `stripped` the
/// lines that are empty once stripped, per `if line.strip()`.
fn dropLines(gpa: Allocator, s: []const u8, comptime stripped: bool) ![]const u8 {
    var kept: List = .empty;
    for (try splitLines(gpa, s)) |line| {
        if ((if (stripped) strip(line) else line).len != 0) try kept.append(gpa, line);
    }
    return joinLines(gpa, kept.items);
}

fn readFile(gpa: Allocator, io: std.Io, path: []const u8) ![]u8 {
    return Dir.cwd().readFileAlloc(io, path, gpa, max_file_bytes);
}

/// Names of the `*.cl` files directly in `dir`, sorted. The Python scripts use
/// `glob.glob`, whose order is the filesystem's; sorting instead makes the
/// generated databases byte-for-byte reproducible.
fn listKernels(gpa: Allocator, io: std.Io, path: []const u8, ext: []const u8) ![][]const u8 {
    var dir = try Dir.cwd().openDir(io, path, .{ .iterate = true });
    defer dir.close(io);

    var names: List = .empty;
    var it = dir.iterate();
    while (try it.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.endsWith(u8, entry.name, ext)) continue;
        try names.append(gpa, try gpa.dupe(u8, entry.name));
    }
    std.mem.sort([]const u8, names.items, {}, lessThan);
    return names.toOwnedSlice(gpa);
}

fn lessThan(_: void, a: []const u8, b: []const u8) bool {
    return std.mem.lessThan(u8, a, b);
}

/// Port of `kernels_db_gen.detect_guard_patterns`. Collects the names of include
/// guards -- an `#ifndef X` / `#if !defined(X)` / `#if !defined X` immediately
/// followed by a bodyless `#define X` -- so that no `#undef` is emitted for
/// them; undefining a guard would defeat it once several kernels that inline the
/// same header are batched into one compilation unit.
fn detectGuardPatterns(gpa: Allocator, content: []const u8) !Set {
    var guards: Set = .empty;
    var i: usize = 0;
    while (std.mem.indexOfScalarPos(u8, content, i, '#')) |hash| {
        i = hash + 1;
        var p = skipSpace(content, hash + 1);

        // `ifndef\s+(\w+)`, `if\s+!\s*defined\s*\(\s*(\w+)\s*\)`, or
        // `if\s+!\s*defined\s+(\w+)`.
        var parenthesised = false;
        if (startsWithAt(content, p, "ifndef")) {
            p = skipSpace1(content, p + "ifndef".len) orelse continue;
        } else if (startsWithAt(content, p, "if")) {
            p = skipSpace1(content, p + "if".len) orelse continue;
            if (p >= content.len or content[p] != '!') continue;
            p = skipSpace(content, p + 1);
            if (!startsWithAt(content, p, "defined")) continue;
            p += "defined".len;
            const after_defined = skipSpace(content, p);
            if (after_defined < content.len and content[after_defined] == '(') {
                parenthesised = true;
                p = skipSpace(content, after_defined + 1);
            } else {
                p = skipSpace1(content, p) orelse continue;
            }
        } else continue;

        const name_end = skipWord(content, p);
        if (name_end == p) continue;
        const first = content[p..name_end];
        p = name_end;

        if (parenthesised) {
            p = skipSpace(content, p);
            if (p >= content.len or content[p] != ')') continue;
            p += 1;
        }

        // `\s*\n\s*#\s*define\s+(\w+)\s*\n`
        const line_break = spaceRunThroughNewline(content, p) orelse continue;
        p = skipSpace(content, line_break);
        if (p >= content.len or content[p] != '#') continue;
        p = skipSpace(content, p + 1);
        if (!startsWithAt(content, p, "define")) continue;
        p = skipSpace1(content, p + "define".len) orelse continue;

        const second_end = skipWord(content, p);
        if (second_end == p) continue;
        const second = content[p..second_end];

        const tail = spaceRunThroughNewline(content, second_end) orelse continue;
        if (std.mem.eql(u8, first, second)) try guards.put(gpa, first, {});
        i = tail;
    }
    return guards;
}

// ---------------------------------------------------------------------------
// primitive_db_gen.py
// ---------------------------------------------------------------------------

const banner = "// This file is autogenerated by tools/cl_kernel_db.zig, a port of " ++
    "primitive_db_gen.py; all changes to this file will be undone\n\n";

/// Chunk limits shared by both of this script's emitters. A raw string literal
/// has an implementation-defined length limit, so long kernels are split across
/// several concatenated literals.
const max_lines = 200;
const max_characters = 16350;

const PrimitiveDb = struct {
    gpa: Allocator,
    io: std.Io,
    /// Absolute path of the `cl_kernels` directory, without a trailing slash.
    kernels_dir: []const u8,
    /// Batch header file names in dependency order.
    batch_headers: [][]const u8 = &.{},
    /// Per origin kernel, every file already inlined into it, in the order they
    /// were inlined. Mirrors `self.include_files`.
    include_files: std.StringArrayHashMapUnmanaged(Set) = .empty,

    const Self = @This();

    /// Port of `find_and_set_batch_headers`. Batch headers hold the macros the
    /// runtime jitter depends on; they are emitted once per batch rather than
    /// inlined into each kernel, so they need to come out in dependency order.
    fn findAndSetBatchHeaders(self: *Self) !void {
        const gpa = self.gpa;
        const dir = try std.fs.path.join(gpa, &.{ self.kernels_dir, "include", "batch_headers" });
        const names = try listKernels(gpa, self.io, dir, ".cl");

        var deps: std.StringArrayHashMapUnmanaged(Set) = .empty;
        for (names) |name| {
            var set: Set = .empty;
            try set.put(gpa, name, {});
            const path = try std.fs.path.join(gpa, &.{ dir, name });
            for (try splitLines(gpa, try readFile(gpa, self.io, path))) |line| {
                if (!std.mem.startsWith(u8, line, "#include")) continue;
                try set.put(gpa, try includeTarget(line), {});
            }
            try deps.put(gpa, name, set);
        }

        var ordered: List = .empty;
        var visiting: Set = .empty;
        var done: Set = .empty;
        for (names) |name| try topologicalSort(gpa, name, deps, &visiting, &done, &ordered);
        self.batch_headers = ordered.items;
    }

    /// Emits `name` after everything it includes, skipping names that are not
    /// batch headers themselves and breaking cycles the way the `stack` check in
    /// `primitive_db_gen.topological_sort` does.
    fn topologicalSort(
        gpa: Allocator,
        name: []const u8,
        deps: std.StringArrayHashMapUnmanaged(Set),
        visiting: *Set,
        done: *Set,
        out: *List,
    ) !void {
        if (done.contains(name) or visiting.contains(name)) return;
        try visiting.put(gpa, name, {});
        for (deps.get(name).?.keys()) |dep| {
            if (!deps.contains(dep)) continue;
            try topologicalSort(gpa, dep, deps, visiting, done, out);
        }
        _ = visiting.orderedRemove(name);
        try done.put(gpa, name, {});
        try out.append(gpa, name);
    }

    /// Port of `append_file_content`: inlines every non-batch-header include,
    /// once per kernel unless the kernel opted out with
    /// `#pragma disable_includes_optimization`.
    fn appendFileContent(self: *Self, path: []const u8, origin: []const u8) ![]const u8 {
        const gpa = self.gpa;
        var res: Buf = .empty;
        var optimize_includes = true;

        for (try splitLines(gpa, try readFile(gpa, self.io, path))) |line| {
            if (std.mem.startsWith(u8, line, "#pragma")) {
                if (std.mem.indexOf(u8, line, "enable_includes_optimization") != null) {
                    optimize_includes = true;
                } else if (std.mem.indexOf(u8, line, "disable_includes_optimization") != null) {
                    optimize_includes = false;
                }
            }
            if (std.mem.startsWith(u8, line, "#include")) {
                const name = try includeTarget(line);
                if (self.isBatchHeader(std.fs.path.basename(name))) continue;

                const full = try std.fs.path.resolve(gpa, &.{ std.fs.path.dirname(path).?, name });
                const inlined = self.include_files.getPtr(origin).?;
                if (!inlined.contains(full) or !optimize_includes) {
                    try inlined.put(gpa, full, {});
                    try res.appendSlice(gpa, try self.appendFileContent(full, origin));
                    try res.append(gpa, '\n');
                }
                continue;
            }
            try res.appendSlice(gpa, rstrip(line));
            try res.append(gpa, '\n');
        }

        // The origin also goes through `reduce_macros`, which is meant to drop
        // `#define`s that nothing uses. Its liveness test scans the `#define`
        // line itself, which always names the macro, so it never drops a line;
        // its rstrip-and-rejoin has already happened above, and the extra
        // trailing newline it adds is dropped again by `postProcessSources`.
        return res.items;
    }

    fn isBatchHeader(self: Self, name: []const u8) bool {
        for (self.batch_headers) |header| {
            if (std.mem.eql(u8, header, name)) return true;
        }
        return false;
    }

    /// Port of `append_undefs`: every macro a kernel or its includes define is
    /// undefined again at the end, so that batching kernels into one compilation
    /// unit cannot leak macros between them. Include guards are left alone.
    fn appendUndefs(self: *Self, path: []const u8) ![]const u8 {
        const gpa = self.gpa;
        var res: Buf = .empty;

        const content = try readFile(gpa, self.io, path);
        const guards = try detectGuardPatterns(gpa, content);

        for (try splitLines(gpa, content)) |line| {
            if (std.mem.indexOf(u8, line, "#define") != null) {
                try emitUndef(gpa, &res, guards, try spaceField(strip(line), 1));
            }
            if (std.mem.indexOf(u8, line, "# define") != null) {
                try emitUndef(gpa, &res, guards, try spaceField(strip(line), 2));
            }
        }

        if (self.include_files.get(path)) |inlined| {
            for (inlined.keys()) |include| {
                try res.appendSlice(gpa, try self.appendUndefs(include));
            }
        }
        return res.items;
    }

    /// Port of `kernel_file_to_str`.
    fn kernelFileToStr(self: *Self, path: []const u8) ![]const u8 {
        const gpa = self.gpa;
        try self.include_files.put(gpa, path, .empty);

        var res: Buf = .empty;
        try res.print(gpa, "{{\"{s}\",\n(std::string) R\"__krnl(\n", .{std.fs.path.stem(path)});

        var body: Buf = .empty;
        try body.appendSlice(gpa, try self.appendFileContent(path, path));
        try body.appendSlice(gpa, try self.appendUndefs(path));
        const content = try postProcessSources(gpa, body.items);

        var characters: usize = 1;
        for (try splitRaw(gpa, content), 0..) |line, i| {
            if ((i + 1) % max_lines == 0 or characters + line.len + 1 > max_characters) {
                try res.appendSlice(gpa, ")__krnl\"\n + R\"__krnl(");
                characters = 0;
            }
            try res.appendSlice(gpa, line);
            try res.append(gpa, '\n');
            characters += line.len + 1;
        }
        try res.appendSlice(gpa, ")__krnl\"},\n\n");
        return res.items;
    }

    /// Port of `batch_headers_to_str`. Note that the character budget is not
    /// reset between headers, matching the original, and that it measures lines
    /// as `readlines()` yields them, with their line break.
    fn batchHeadersToStr(self: *Self) ![]const u8 {
        const gpa = self.gpa;
        var res: Buf = .empty;
        var characters: usize = 1;

        for (self.batch_headers) |header| {
            try res.print(gpa, "{{\"{s}\",\n(std::string) R\"-(\n", .{std.fs.path.stem(header)});
            const path = try std.fs.path.join(
                gpa,
                &.{ self.kernels_dir, "include", "batch_headers", header },
            );
            for (try splitLines(gpa, try readFile(gpa, self.io, path)), 0..) |line, i| {
                if (std.mem.startsWith(u8, line, "#include")) continue;
                if ((i + 1) % max_lines == 0 or characters + line.len + 2 > max_characters) {
                    try res.appendSlice(gpa, ")-\"\n + (std::string) R\"-(");
                    characters = 0;
                }
                try res.appendSlice(gpa, rstrip(line));
                try res.append(gpa, '\n');
                characters += line.len + 2;
            }
            try res.appendSlice(gpa, ")-\"},\n\n");
        }
        return postProcessSources(gpa, res.items);
    }
};

/// `line.strip().split('"')[1].strip()`: the text between the first two quotes.
fn includeTarget(line: []const u8) ![]const u8 {
    const trimmed = strip(line);
    const open = std.mem.indexOfScalar(u8, trimmed, '"') orelse return error.MalformedInclude;
    const rest = trimmed[open + 1 ..];
    const close = std.mem.indexOfScalar(u8, rest, '"') orelse return error.MalformedInclude;
    return strip(rest[0..close]);
}

/// `s.split(" ")[n]`: split on single spaces, without collapsing runs.
fn spaceField(s: []const u8, n: usize) ![]const u8 {
    var it = std.mem.splitScalar(u8, s, ' ');
    var i: usize = 0;
    while (it.next()) |field| : (i += 1) {
        if (i == n) return field;
    }
    return error.MissingField;
}

fn emitUndef(gpa: Allocator, res: *Buf, guards: Set, field: []const u8) !void {
    const name = field[0 .. std.mem.indexOfScalar(u8, field, '(') orelse field.len];
    if (guards.contains(name)) return;
    try res.print(gpa, "#ifdef {s}\n#undef {s}\n#endif\n", .{ name, name });
}

/// Port of `post_process_sources`: strips comments, blank lines, line
/// continuations and repeated spaces.
fn postProcessSources(gpa: Allocator, content: []const u8) ![]const u8 {
    var out: Buf = .empty;

    // `(^)?[^\S\n]*/(?:\*(.*?)\*/[^\S\n]*|/[^\n]*)($)?` with a replacement that
    // keeps a block comment's line structure: a comment touching either end of
    // its line disappears along with the whitespace leading up to it, one that
    // sits mid-line collapses to a single space, and one that spans lines
    // collapses to a newline.
    var i: usize = 0;
    while (i < content.len) {
        const at_line_start = i == 0 or content[i - 1] == '\n';
        const slash = skipHSpace(content, i);

        if (slash + 1 < content.len and content[slash] == '/') {
            if (content[slash + 1] == '*') {
                if (std.mem.indexOfPos(u8, content, slash + 2, "*/")) |close| {
                    const body = content[slash + 2 .. close];
                    const after = skipHSpace(content, close + 2);
                    const at_line_end = after == content.len or content[after] == '\n';
                    if (!at_line_start and !at_line_end) {
                        const spans_lines = std.mem.indexOfScalar(u8, body, '\n') != null;
                        try out.appendSlice(gpa, if (spans_lines) "\n" else " ");
                    }
                    i = after;
                    continue;
                }
            } else if (content[slash + 1] == '/') {
                i = std.mem.indexOfScalarPos(u8, content, slash + 2, '\n') orelse content.len;
                continue;
            }
        }

        try out.append(gpa, content[i]);
        i += 1;
    }

    // Drop empty lines, fold line continuations, collapse runs of spaces.
    const compact = try dropLines(gpa, out.items, false);
    const folded = try std.mem.replaceOwned(u8, gpa, compact, "\\\n", "");
    return std.mem.collapseRepeats(u8, folded, ' ');
}

fn generatePrimitiveDb(
    gpa: Allocator,
    io: std.Io,
    kernels_dir: []const u8,
    out_dir: []const u8,
) !void {
    var db: PrimitiveDb = .{
        .gpa = gpa,
        .io = io,
        .kernels_dir = try std.fs.path.resolve(gpa, &.{kernels_dir}),
    };
    try db.findAndSetBatchHeaders();

    var primitives: Buf = .empty;
    try primitives.appendSlice(gpa, banner);
    for (try listKernels(gpa, io, db.kernels_dir, ".cl")) |name| {
        const path = try std.fs.path.join(gpa, &.{ db.kernels_dir, name });
        try primitives.appendSlice(gpa, try db.kernelFileToStr(path));
    }
    try writeOutput(io, out_dir, "ks_primitive_db.inc", primitives.items);

    var headers: Buf = .empty;
    try headers.appendSlice(gpa, banner);
    try headers.appendSlice(gpa, try db.batchHeadersToStr());
    try writeOutput(io, out_dir, "ks_primitive_db_batch_headers.inc", headers.items);
}

// ---------------------------------------------------------------------------
// kernels_db_gen.py
// ---------------------------------------------------------------------------

/// The characters `minimize_code` closes whitespace up around.
fn isSqueezed(c: u8) bool {
    return switch (c) {
        '{', '}', '=', ';', ',', '+', '-', '<', '>', '!', '&', '|', '%', '#' => true,
        else => false,
    };
}

/// End of the `[^()]*` run starting at `i`.
fn skipNonParen(s: []const u8, i: usize) usize {
    return std.mem.findAnyPos(u8, s, i, "()") orelse s.len;
}

/// Port of `minimize_code`: strips comments and squeezes the kernel down to one
/// statement per line with no redundant whitespace.
fn minimizeCode(gpa: Allocator, content: []const u8) ![]const u8 {
    // `//.*` -> '' (no DOTALL, so it stops at the line break).
    var no_line_comments: Buf = .empty;
    var i: usize = 0;
    while (i < content.len) {
        if (content[i] == '/' and i + 1 < content.len and content[i + 1] == '/') {
            i = std.mem.indexOfScalarPos(u8, content, i + 2, '\n') orelse content.len;
            continue;
        }
        try no_line_comments.append(gpa, content[i]);
        i += 1;
    }

    // `/\*.*?\*/` -> '' (DOTALL, non-greedy).
    const one = no_line_comments.items;
    var no_comments: Buf = .empty;
    i = 0;
    while (i < one.len) {
        if (one[i] == '/' and i + 1 < one.len and one[i + 1] == '*') {
            if (std.mem.indexOfPos(u8, one, i + 2, "*/")) |close| {
                i = close + 2;
                continue;
            }
        }
        try no_comments.append(gpa, one[i]);
        i += 1;
    }

    // Strip each line, drop the empty ones, collapse `\s+` to a single space,
    // then close whitespace up around the operators in `isSqueezed`.
    var kept: List = .empty;
    for (try splitLines(gpa, no_comments.items)) |raw| {
        const line = strip(raw);
        if (line.len == 0) continue;

        var squeezed: Buf = .empty;
        var j: usize = 0;
        while (j < line.len) {
            const sym = skipSpace(line, j);
            if (sym < line.len and isSqueezed(line[sym])) {
                try squeezed.append(gpa, line[sym]);
                j = skipSpace(line, sym + 1);
                continue;
            }
            if (isSpace(line[j])) {
                try squeezed.append(gpa, ' ');
                j = skipSpace(line, j);
                continue;
            }
            try squeezed.append(gpa, line[j]);
            j += 1;
        }
        try kept.append(gpa, squeezed.items);
    }

    // `\\\s*\n` -> '': fold multi-line macros onto one line.
    const joined = try joinLines(gpa, kept.items);
    var folded: Buf = .empty;
    i = 0;
    while (i < joined.len) {
        if (joined[i] == '\\') {
            if (spaceRunThroughNewline(joined, i + 1)) |end| {
                i = end;
                continue;
            }
        }
        try folded.append(gpa, joined[i]);
        i += 1;
    }
    return folded.items;
}

const Include = struct {
    name: []const u8,
    no_opt: bool,
    end: usize,
};

/// `#include\s+"([^"]+)"(\s+\[\[no_opt\]\])?` anchored at `i`.
fn matchInclude(s: []const u8, i: usize) ?Include {
    if (!startsWithAt(s, i, "#include")) return null;
    const p = skipSpace1(s, i + "#include".len) orelse return null;
    if (p >= s.len or s[p] != '"') return null;
    const close = std.mem.indexOfScalarPos(u8, s, p + 1, '"') orelse return null;
    if (close == p + 1) return null;

    const name = s[p + 1 .. close];
    var end = close + 1;
    var no_opt = false;
    const attr = skipSpace(s, end);
    if (attr > end and startsWithAt(s, attr, "[[no_opt]]")) {
        no_opt = true;
        end = attr + "[[no_opt]]".len;
    }
    return .{ .name = name, .no_opt = no_opt, .end = end };
}

/// Port of `process_includes`: inlines every include that resolves under one of
/// `include_paths`, once, unless it carries `[[no_opt]]`. Batch headers stay as
/// `#include` lines hoisted to the top; the runtime resolves those itself.
fn processIncludes(
    gpa: Allocator,
    io: std.Io,
    content: []const u8,
    include_paths: []const []const u8,
    processed: *Set,
) error{ OutOfMemory, ReadFailed }![]const u8 {
    var batch_headers: List = .empty;
    var out: Buf = .empty;

    var i: usize = 0;
    while (i < content.len) {
        const include = matchInclude(content, i) orelse {
            try out.append(gpa, content[i]);
            i += 1;
            continue;
        };
        i = include.end;

        if (processed.contains(include.name) and !include.no_opt) continue;

        if (std.mem.indexOf(u8, include.name, "batch_headers") != null) {
            try processed.put(gpa, include.name, {});
            const line = try std.fmt.allocPrint(gpa, "#include \"{s}\"", .{include.name});
            try batch_headers.insert(gpa, 0, line);
            continue;
        }

        for (include_paths) |dir| {
            const full = try std.fs.path.join(gpa, &.{ dir, include.name });
            const nested = readFile(gpa, io, full) catch continue;
            try processed.put(gpa, include.name, {});
            try out.appendSlice(gpa, try processIncludes(gpa, io, nested, include_paths, processed));
            break;
        }
    }

    var res: Buf = .empty;
    try res.appendSlice(gpa, try joinLines(gpa, batch_headers.items));
    try res.append(gpa, '\n');
    try res.appendSlice(gpa, strip(out.items));
    return res.items;
}

const CatMatch = struct {
    start: usize,
    end: usize,
    body: []const u8,
};

/// The next `CAT(...)` at or after `from`. Greedy, without backtracking -- the
/// inputs never need it. With `nested` this is
/// `CAT\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)`, whose arguments may contain any
/// number of parenthesised groups; without it,
/// `CAT\s*\(([^()]*|(?:[^()]*\([^\)]*\))[^()]*)\)`, the variant used when
/// expanding concatenations, which admits one group that may itself hold a `(`.
fn findCat(s: []const u8, from: usize, comptime nested: bool) ?CatMatch {
    var i = from;
    while (std.mem.indexOfPos(u8, s, i, "CAT")) |at| {
        i = at + 1;
        const open = skipSpace(s, at + 3);
        if (open >= s.len or s[open] != '(') continue;

        var p = skipNonParen(s, open + 1);
        if (nested) {
            while (p < s.len and s[p] == '(') {
                const close = skipNonParen(s, p + 1);
                if (close >= s.len or s[close] != ')') break;
                p = skipNonParen(s, close + 1);
            }
        } else if (p < s.len and s[p] == '(') {
            const inner = std.mem.indexOfScalarPos(u8, s, p + 1, ')') orelse continue;
            p = skipNonParen(s, inner + 1);
        }
        if (p >= s.len or s[p] != ')') continue;
        return .{ .start = at, .end = p + 1, .body = s[open + 1 .. p] };
    }
    return null;
}

/// `re.split(r'\s*,\s*', s)`.
fn splitOnCommas(gpa: Allocator, s: []const u8) ![][]const u8 {
    var parts: List = .empty;
    var start: usize = 0;
    var i: usize = 0;
    while (i < s.len) : (i += 1) {
        if (s[i] != ',') continue;
        try parts.append(gpa, rstrip(s[start..i]));
        i = skipSpace(s, i + 1) - 1;
        start = i + 1;
    }
    try parts.append(gpa, s[start..]);
    return parts.toOwnedSlice(gpa);
}

/// Port of `expand_cat_expression`: rewrites `CAT(a, b)` to `ab` until no
/// concatenation is left, so that macros reached only through `CAT` still count
/// as used. The Python version spins forever if a `CAT` is written with a space
/// before its parenthesis, because the literal replacement it builds no longer
/// matches; this stops instead.
fn expandCat(gpa: Allocator, expression: []const u8) ![]const u8 {
    var current = expression;
    while (true) {
        var changed = false;
        var i: usize = 0;
        while (findCat(current, i, false)) |match| {
            i = match.end;
            const parts = try splitOnCommas(gpa, match.body);
            const expanded = try std.mem.concat(gpa, u8, parts);
            const needle = try std.fmt.allocPrint(gpa, "CAT({s})", .{match.body});
            if (std.mem.indexOf(u8, current, needle) == null) continue;
            current = try std.mem.replaceOwned(u8, gpa, current, needle, expanded);
            changed = true;
            i = 0;
        }
        if (!changed) return current;
    }
}

/// Port of `_check_cat_usage`: reports whether a `CAT` on this line could build
/// the macro's name out of its arguments.
fn checkCatUsage(gpa: Allocator, macro: []const u8, line: []const u8) !bool {
    var i: usize = 0;
    while (findCat(line, i, true)) |match| {
        i = match.end;
        const parts = try splitOnCommas(gpa, match.body);
        if (parts.len < 2) continue;

        var potential: Buf = .empty;
        for (parts) |part| try potential.appendSlice(gpa, strip(part));
        if (std.mem.indexOf(u8, potential.items, macro) != null) return true;
        for (parts) |part| {
            if (std.mem.indexOf(u8, macro, strip(part)) != null) return true;
        }
    }
    return false;
}

/// `\bmacro\b` in `line`.
fn mentionsMacro(macro: []const u8, line: []const u8) bool {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, line, i, macro)) |at| {
        i = at + 1;
        if (at > 0 and isWord(line[at - 1])) continue;
        const after = at + macro.len;
        if (after < line.len and isWord(line[after])) continue;
        return true;
    }
    return false;
}

/// `^\s*#\s*(define|undef)\s+macro\b` anchored at the start of `line`.
fn definesMacro(macro: []const u8, line: []const u8) bool {
    var p = skipSpace(line, 0);
    if (p >= line.len or line[p] != '#') return false;
    p = skipSpace(line, p + 1);
    if (startsWithAt(line, p, "define")) {
        p += "define".len;
    } else if (startsWithAt(line, p, "undef")) {
        p += "undef".len;
    } else return false;

    p = skipSpace1(line, p) orelse return false;
    if (!startsWithAt(line, p, macro)) return false;
    const end = p + macro.len;
    return end == line.len or !isWord(line[end]);
}

/// Port of `found_potential_macro_user`.
fn foundPotentialMacroUser(gpa: Allocator, macro: []const u8, content: []const u8) !bool {
    for (try splitRaw(gpa, content)) |line| {
        if (definesMacro(macro, line)) continue;
        if (mentionsMacro(macro, line)) return true;
        if (std.mem.indexOf(u8, line, "CAT") != null and try checkCatUsage(gpa, macro, line)) {
            return true;
        }
    }
    return false;
}

const Directive = struct {
    name: []const u8,
    /// The rest of the line after the name, `\s*(.*)`.
    body: []const u8,
    /// Where the name ends.
    end: usize,
};

/// `^<keyword>\s+(\w+)` anchored at the start of `s`.
fn parseDirective(s: []const u8, comptime keyword: []const u8) ?Directive {
    if (!std.mem.startsWith(u8, s, keyword)) return null;
    const p = skipSpace1(s, keyword.len) orelse return null;
    const end = skipWord(s, p);
    if (end == p) return null;
    return .{ .name = s[p..end], .body = s[skipSpace(s, end)..], .end = end };
}

/// Port of `remove_unused_macros`: drops `#define`s (and their `#undef`s) that
/// nothing in the kernel references, directly or through a `CAT`.
fn removeUnusedMacros(gpa: Allocator, content: []const u8) ![]const u8 {
    var macros: std.StringArrayHashMapUnmanaged([]const u8) = .empty;
    for (try splitLines(gpa, content)) |line| {
        if (parseDirective(line, "#define")) |define| try macros.put(gpa, define.name, define.body);
    }

    var used: Set = .empty;
    for (macros.keys()) |name| {
        if (try foundPotentialMacroUser(gpa, name, content)) try used.put(gpa, name, {});
    }
    // A macro reached only as the result of a concatenation counts as used.
    for (macros.values()) |body| {
        const expanded = try expandCat(gpa, body);
        if (macros.contains(expanded)) try used.put(gpa, expanded, {});
    }
    var i: usize = 0;
    while (findCat(content, i, false)) |match| {
        i = match.end;
        const expanded = try expandCat(gpa, content[match.start..match.end]);
        if (macros.contains(expanded)) try used.put(gpa, expanded, {});
    }

    var out: List = .empty;
    for (try splitLines(gpa, content)) |line| {
        if (parseDirective(line, "#define")) |define| {
            const unused = macros.contains(define.name) and !used.contains(define.name);
            try out.append(gpa, if (unused) "" else line);
            continue;
        }
        if (parseDirective(line, "#undef")) |undef| {
            const unused = macros.contains(undef.name) and !used.contains(undef.name);
            try out.append(gpa, if (unused) line[undef.end..] else line);
            continue;
        }
        try out.append(gpa, line);
    }
    return dropLines(gpa, try joinLines(gpa, out.items), true);
}

/// `#undef\s+name\s*(?:\n|$)` anywhere in `content`.
fn hasUndef(content: []const u8, name: []const u8) bool {
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, content, i, "#undef")) |at| {
        i = at + 1;
        const undef = parseDirective(content[at..], "#undef") orelse continue;
        if (!std.mem.eql(u8, undef.name, name)) continue;

        const end = at + undef.end;
        const run = skipSpace(content, end);
        if (run == content.len) return true;
        if (std.mem.indexOfScalar(u8, content[end..run], '\n') != null) return true;
    }
    return false;
}

/// Port of `add_missing_undefs`: balances every remaining `#define` with an
/// `#undef`, so batched kernels cannot leak macros into each other.
fn addMissingUndefs(gpa: Allocator, content: []const u8) ![]const u8 {
    var defines: Set = .empty;
    var i: usize = 0;
    while (std.mem.indexOfPos(u8, content, i, "#define")) |at| {
        i = at + 1;
        const define = parseDirective(content[at..], "#define") orelse continue;
        try defines.put(gpa, define.name, {});
        i = at + define.end;
    }

    var guards = try detectGuardPatterns(gpa, content);
    var undefs: List = .empty;
    for (defines.keys()) |name| {
        if (guards.contains(name)) continue;
        if (hasUndef(content, name)) continue;
        try undefs.append(gpa, try std.fmt.allocPrint(gpa, "#undef {s}", .{name}));
    }
    if (undefs.items.len == 0) return content;

    std.mem.sort([]const u8, undefs.items, {}, lessThan);
    var out: Buf = .empty;
    try out.appendSlice(gpa, content);
    try out.append(gpa, '\n');
    try out.appendSlice(gpa, try joinLines(gpa, undefs.items));
    return out.items;
}

/// Port of `process_file`.
fn processFile(
    gpa: Allocator,
    io: std.Io,
    path: []const u8,
    include_dirs: []const []const u8,
    is_batch_header: bool,
) ![]const u8 {
    const max_length = 5000;

    var content = try minimizeCode(gpa, try readFile(gpa, io, path));
    if (!is_batch_header) {
        var processed: Set = .empty;
        content = try processIncludes(gpa, io, content, include_dirs, &processed);
        content = try minimizeCode(gpa, content);
        content = try removeUnusedMacros(gpa, content);
        content = try addMissingUndefs(gpa, content);
    }

    // One raw string literal per `max_length` chunk; an empty kernel still
    // gets one.
    var literals: List = .empty;
    var i: usize = 0;
    while (true) : (i += max_length) {
        const chunk = content[i..@min(i + max_length, content.len)];
        try literals.append(gpa, try std.fmt.allocPrint(gpa, "R\"__krnl({s})__krnl\"", .{chunk}));
        if (i + max_length >= content.len) break;
    }

    return std.fmt.allocPrint(
        gpa,
        "std::make_pair<std::string_view, std::string_view>(\"{s}\", {s}),\n",
        .{ std.fs.path.stem(path), try joinLines(gpa, literals.items) },
    );
}

fn generateOclV2(
    gpa: Allocator,
    io: std.Io,
    kernels_dir: []const u8,
    headers_dir: []const u8,
    out_dir: []const u8,
) !void {
    const kernels = try std.fs.path.resolve(gpa, &.{kernels_dir});
    const headers = try std.fs.path.resolve(gpa, &.{headers_dir});
    const batch_headers = try std.fs.path.join(gpa, &.{ headers, "batch_headers" });
    const include_dirs: []const []const u8 = &.{
        kernels,
        headers,
        try std.fs.path.join(gpa, &.{ headers, ".." }),
        batch_headers,
    };

    var sources: Buf = .empty;
    for (try listKernels(gpa, io, kernels, ".cl")) |name| {
        const path = try std.fs.path.join(gpa, &.{ kernels, name });
        try sources.appendSlice(gpa, try processFile(gpa, io, path, include_dirs, false));
    }
    try writeOutput(io, out_dir, "gpu_ocl_kernel_sources.inc", sources.items);

    var headers_out: Buf = .empty;
    for (try listKernels(gpa, io, batch_headers, ".cl")) |name| {
        const path = try std.fs.path.join(gpa, &.{ batch_headers, name });
        try headers_out.appendSlice(gpa, try processFile(gpa, io, path, &.{}, true));
    }
    try writeOutput(io, out_dir, "gpu_ocl_kernel_headers.inc", headers_out.items);
}
