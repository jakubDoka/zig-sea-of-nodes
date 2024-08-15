lexer: Lexer,
cur: Token = undefined,
son: Son = .{},
prev_control: Id = undefined,
gpa: std.mem.Allocator,

gvn: gvn.Map = .{},
ctx: struct {
    vars: std.ArrayListUnmanaged(Var) = .{},
} = .{},

const std = @import("std");
const debug = @import("builtin").mode == .Debug;
const Lexer = @import("Lexer.zig");
const Token = Lexer.Token;
const Son = @import("Son.zig");
const Kind = Son.Kind;
const Fn = Son.Fn;
const Node = Son.Node;
const Id = Son.Id;
const Parser = @This();
const Error = std.mem.Allocator.Error;

const gvn = struct {
    const Map = std.HashMapUnmanaged(Key, void, Context, 80);

    const Key = struct {
        id: Id,
        hash: u64,
    };

    fn hash(kind: Kind, inputs: anytype) u64 {
        var hasher = std.hash.Fnv1a_64.init();
        std.hash.autoHash(&hasher, kind);
        std.hash.autoHash(&hasher, inputs);
        return hasher.final();
    }

    fn hashInput(kind: Kind, inputs: Son.Inputs) u64 {
        var hasher = std.hash.Fnv1a_64.init();
        std.hash.autoHash(&hasher, kind);
        switch (kind) {
            inline else => |r| std.hash.autoHash(&hasher, @field(inputs, r.inputPayloadName())),
        }
        return hasher.final();
    }

    fn KeyContext(comptime P: type) type {
        return struct {
            son: *const Son,
            inputs: P,

            pub fn hash(_: @This(), val: Key) u64 {
                return val.hash;
            }

            pub fn eql(self: @This(), left: Key, right: Key) bool {
                if (left.id.kind() != right.id.kind()) return false;
                if (left.hash != right.hash) return false;

                const right_inputs = self.son.get(right.id).inputs;
                const name = comptime Son.Inputs.nameForPayload(P);
                return std.meta.eql(self.inputs, @field(right_inputs, name));
            }
        };
    }

    const Context = struct {
        pub fn hash(_: Context, val: Key) u64 {
            return val.hash;
        }

        pub fn eql(_: Context, left: Key, right: Key) bool {
            return left.id.eql(right.id);
        }
    };
};

const Var = struct {
    offset: u32,
    value: Id,

    const arg_sentinel = std.math.maxInt(u32);
};

fn dbg(any: anytype) @TypeOf(any) {
    std.debug.print("{any}\n", .{any});
    return any;
}

pub fn assertIntegirty(self: Parser, entry: Id) !void {
    var leaked = std.ArrayList(Id).init(self.gpa);
    defer leaked.deinit();
    try self.son.collectLeakedIds(entry, &leaked);
    for (leaked.items) |it| {
        switch (it.kind()) {
            inline else => |k| {
                const payload = @field(self.son.get(it).inputs, k.inputPayloadName());
                std.debug.print("{}: {any}\n", .{ it, payload });
            },
        }
    }
    try std.testing.expectEqualSlices(Id, &.{}, leaked.items);
}

pub fn deinit(self: *Parser) void {
    inline for (@typeInfo(@TypeOf(self.ctx)).Struct.fields) |field| {
        std.debug.assert(@field(self.ctx, field.name).items.len == 0);
        @field(self.ctx, field.name).deinit(self.gpa);
    }
    self.gvn.deinit(self.gpa);
    self.* = undefined;
}

fn next(self: *Parser) !?Id {
    if (self.cur.lexeme == .Eof) return null;
    return try self.nextExpr();
}

fn nextExpr(self: *Parser) Error!Id {
    return self.nextBinOp(try self.nextUnit(), 254);
}

fn nextBinOp(self: *Parser, lhs: Id, prec: u8) !Id {
    var acc = lhs;
    while (true) {
        const next_prec = self.cur.lexeme.prec();
        if (next_prec >= prec) break;

        try self.son.getPtr(acc).refs.append(self.gpa, &self.son.slices, .{});
        const op = self.advance().lexeme;
        const rhs = try self.nextBinOp(try self.nextUnit(), next_prec);
        self.son.getPtr(acc).refs.remove(&self.son.slices, .{});
        switch (op) {
            else => acc = switch (op) {
                inline else => |t| if (@hasField(Kind, "bo" ++ @tagName(t)) and comptime t.isOp())
                    try self.alloc(@field(Kind, "bo" ++ @tagName(t)), .{ .lhs = acc, .rhs = rhs })
                else
                    std.debug.panic("unhandled binary poerator: {s}", .{@tagName(t)}),
            },
        }
    }
    return acc;
}

fn nextUnit(self: *Parser) !Id {
    const token = self.advance();
    switch (token.lexeme) {
        .@"return" => {
            self.prev_control = try self.alloc(.cfg_return, .{
                .cfg = self.prev_control,
                .value = try self.nextExpr(),
            });
            return self.prev_control;
        },
        .Int => return try self.alloc(.const_int, .{
            .value = std.fmt.parseInt(i64, token.view(self.lexer.source), 10) catch
                unreachable,
        }),
        .Ident => {
            if (std.mem.eql(u8, token.view(self.lexer.source), "arg")) {
                return self.ctx.vars.items[0].value;
            }
            unreachable;
        },
        .@"-" => return try self.alloc(.@"uo-", .{ .oper = try self.nextUnit() }),
        .@"(" => {
            const expr = try self.nextExpr();
            std.debug.assert(self.advance().lexeme == .@")");
            return expr;
        },
        else => |e| std.debug.panic("unhandled token: {s}", .{@tagName(e)}),
    }
    return .{};
}

inline fn advance(self: *Parser) Token {
    defer self.cur = self.lexer.next();
    return self.cur;
}

fn allocAny(self: *Parser, kind: Kind, payload: anytype) !Id {
    return switch (kind) {
        inline else => |k| self.alloc(k, payload),
    };
}

fn alloc(self: *Parser, comptime kind: Kind, payload: kind.InputPayload()) !Id {
    if (try self.peephole(kind, payload)) |rpl| {
        const inps = Son.Inputs.idsOfPayload(&payload);
        inline for (inps, 0..) |inp, i| {
            if (std.mem.indexOfScalar(u32, @ptrCast(inps[0..i]), inp.repr()) == null) {
                self.remove(inp);
            }
        }
        return rpl;
    }

    return try self.peepholeAlloc(kind, payload);
}

fn remove(self: *Parser, id: Id) void {
    const nd = self.son.get(id);
    if (nd.refs.len() == 0) {
        std.debug.assert(self.gvn.remove(gvn.Key{
            .id = id,
            .hash = gvn.hashInput(id.kind(), nd.inputs),
        }));
        switch (id.kind()) {
            inline else => |t| {
                const inps = Son.Inputs.idsOf(&nd.inputs, t.inputPayloadName());
                inline for (inps, 0..) |inp, i| {
                    if (std.mem.indexOfScalar(u32, @ptrCast(inps[0..i]), inp.repr()) == null) {
                        self.son.getPtr(inp).refs.remove(&self.son.slices, id);
                        self.remove(inp);
                    }
                }
            },
        }
        self.son.rmeove(id.index);
    }
}

fn getGwn(self: *Parser, comptime kind: Kind, payload: kind.InputPayload()) !gvn.Map.GetOrPutResult {
    const hash = gvn.hash(kind, payload);
    const result = try self.gvn.getOrPutAdapted(
        self.gpa,
        gvn.Key{ .id = Id.invalid(kind), .hash = hash },
        gvn.KeyContext(@TypeOf(payload)){ .son = &self.son, .inputs = payload },
    );
    result.key_ptr.hash = hash;
    return result;
}

fn peephole(self: *Parser, comptime kind: Kind, payload: kind.InputPayload()) !?Id {
    return switch (@TypeOf(payload)) {
        Son.BinOp => self.peepholeBinOp(kind, payload),
        Son.UnOp => self.peepholeUnOp(kind, payload),
        else => null,
    };
}

fn peepholeBinOp(self: *Parser, comptime kind: Kind, abo: Son.BinOp) !?Id {
    var bo, var changed = .{ abo, false };
    var const_lhs = bo.lhs.kind().isConst();
    var const_rhs = bo.rhs.kind().isConst();
    const commutatuive = comptime kind.isCommutative();

    if (const_lhs and const_rhs) return try self.peepholeAlloc(.const_int, .{
        .value = kind.applyBinOp(
            self.son.get(bo.lhs).inputs.const_int.value,
            self.son.get(bo.rhs).inputs.const_int.value,
        ),
    });

    if (commutatuive and (const_lhs or (bo.lhs.index > bo.rhs.index and !const_rhs))) {
        changed = true;
        std.mem.swap(Id, &bo.rhs, &bo.lhs);
        std.mem.swap(bool, &const_rhs, &const_lhs);
    }

    if (commutatuive and bo.lhs.kind() == kind) {
        const lhs = self.son.get(bo.lhs).inputs.bo;
        const const_lhs_rhs = lhs.rhs.kind().isConst();

        if (const_lhs_rhs and const_rhs) return try self.peepholeAlloc(kind, .{
            .lhs = lhs.lhs,
            .rhs = try self.peepholeAlloc(.const_int, .{ .value = kind.applyBinOp(
                self.son.get(lhs.rhs).inputs.const_int.value,
                self.son.get(bo.rhs).inputs.const_int.value,
            ) }),
        });

        unreachable;
    }

    if (bo.lhs.eql(bo.rhs) and !const_lhs) switch (kind) {
        .@"bo+" => return try self.peepholeAlloc(.@"bo*", .{
            .lhs = bo.lhs,
            .rhs = try self.peepholeAlloc(.const_int, .{ .value = 2 }),
        }),
        .@"bo-" => return try self.peepholeAlloc(.const_int, .{ .value = 0 }),
        .@"bo/" => return try self.peepholeAlloc(.const_int, .{ .value = 1 }),
        else => {},
    };

    if (changed) return try self.peepholeAlloc(kind, bo);

    return null;
}

fn peepholeUnOp(self: *Parser, comptime kind: Kind, uo: Son.UnOp) !?Id {
    const const_oper = uo.oper.kind().isConst();

    if (const_oper) return try self.peepholeAlloc(.const_int, .{
        .value = kind.applyUnOp(
            self.son.get(uo.oper).inputs.const_int.value,
        ),
    });

    return null;
}

fn peepholeAlloc(self: *Parser, comptime kind: Kind, payload: kind.InputPayload()) !Id {
    const result = try self.getGwn(kind, payload);
    if (result.found_existing) return result.key_ptr.id;
    result.key_ptr.id = try self.son.add(self.gpa, kind, payload);
    inline for (self.son.get(result.key_ptr.id).inputs.idsOf(kind.inputPayloadName())) |inp| {
        try self.son.getPtr(inp).refs.append(self.gpa, &self.son.slices, result.key_ptr.id);
    }
    return result.key_ptr.id;
}

fn testParse(code: []const u8) !struct { Son, Fn } {
    var parser = Parser{
        .gpa = std.testing.allocator,
        .lexer = Lexer{ .source = code },
    };
    defer parser.deinit();
    errdefer parser.son.deinit(std.testing.allocator);
    parser.cur = parser.lexer.next();
    const entry = try parser.son.add(parser.gpa, .cfg_start, {});
    parser.prev_control = try parser.alloc(.cfg_tuple, .{ .on = entry, .index = 0 });

    try parser.ctx.vars.append(parser.gpa, .{
        .offset = Var.arg_sentinel,
        .value = try parser.alloc(.cfg_tuple, .{ .on = entry, .index = 1 }),
    });
    var last_node = entry;
    while (try parser.next()) |node| last_node = node;
    parser.ctx.vars.items.len = 0;
    try parser.assertIntegirty(entry);
    return .{ parser.son, .{ .entry = entry, .exit = last_node } };
}

fn constCase(exit: i64, code: []const u8) !void {
    var son, const fnc = try testParse(code);
    defer son.deinit(std.testing.allocator);
    const rvl = son.get(fnc.exit).inputs.cfg_return.value;
    const cnst = son.get(rvl).inputs.const_int.value;
    try std.testing.expectEqual(exit, cnst);
}

fn dynCase(code: []const u8, output: *std.ArrayList(u8)) !void {
    const gpa = std.testing.allocator;
    try output.writer().print("CASE: {s}\n", .{code});
    var son, const fnc = try testParse(code);
    defer son.deinit(gpa);
    {
        var fmt = Son.Fmt{ .son = &son, .out = output };
        defer fmt.deinit();
        try fmt.fmt(fnc.entry);
    }
}

fn resolveDynCases(comptime name: []const u8, output: []const u8) !void {
    const gpa = std.testing.allocator;

    const old, const new = .{ "tests/" ++ name ++ ".temp.old.txt", "tests/" ++ name ++ ".temp.new.txt" };

    const update = std.process.getEnvVarOwned(gpa, "PT_UPDATE") catch "";
    defer gpa.free(update);

    if (update.len > 0) {
        try std.fs.cwd().writeFile(.{
            .sub_path = old,
            .data = std.mem.trim(u8, output, "\n"),
        });
        try std.fs.cwd().deleteFile(new);
    } else {
        try std.fs.cwd().writeFile(.{
            .sub_path = new,
            .data = std.mem.trim(u8, output, "\n"),
        });
        const err = runDiff(gpa, old, new) catch |e| switch (e) {
            error.FileNotFound => {
                std.debug.print("\nNEW_OUTPUT:\n{s}", .{output});
            },
            else => e,
        };

        try err;
    }
}

pub fn runDiff(gpa: std.mem.Allocator, old: []const u8, new: []const u8) !void {
    var child = std.process.Child.init(&.{ "diff", "-y", old, new }, gpa);
    child.stdout_behavior = .Pipe;
    child.stderr_behavior = .Pipe;

    try child.spawn();

    const stdout = try child.stderr.?.readToEndAlloc(gpa, 1024 * 10);
    defer gpa.free(stdout);
    const stderr = try child.stdout.?.readToEndAlloc(gpa, 1024 * 10);
    defer gpa.free(stderr);

    const exit = (try child.wait()).Exited;
    if (exit != 0) {
        const new_data = try std.fs.cwd().readFileAlloc(gpa, new, 1024 * 1024);
        defer gpa.free(new_data);
        const old_data = try std.fs.cwd().readFileAlloc(gpa, old, 1024 * 1024);
        defer gpa.free(old_data);
        const new_line_count: isize = @intCast(std.mem.count(u8, new_data, "\n"));
        const old_line_count: isize = @intCast(std.mem.count(u8, old_data, "\n"));
        std.debug.print("line count change: {d}\n", .{new_line_count - old_line_count});
        if (stdout.len > 0) std.debug.print("stdout:\n{s}", .{stdout});
        if (stderr.len > 0) std.debug.print("stderr:\n{s}", .{stderr});
    }
    try std.testing.expectEqual(0, exit);
}

test "math" {
    try constCase(0, "return 1 - 1");
    try constCase(0, "return 1 + -1");
    try constCase(0, "return 1 + 4 / 4 - 2 * 1");
    try constCase(0, "return (1 + arg + 2) - (arg + 3)");

    var output = std.ArrayList(u8).init(std.testing.allocator);
    defer output.deinit();
    try dynCase("return 1 + arg + 2", &output);
    try dynCase("return 1 + (arg + 2)", &output);
    try resolveDynCases("with argument", output.items);
}
