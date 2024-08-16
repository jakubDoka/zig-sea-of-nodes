lexer: Lexer,
cur: Token = undefined,
son: Son = .{},
control: Id = undefined,
gpa: std.mem.Allocator,
last_var: u32 = undefined,

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
        try self.lockNode(acc);
        const op = self.advance().lexeme;
        const last_var = self.last_var;
        const rhs = try self.nextBinOp(try self.nextUnit(), next_prec);
        self.unlockNode(acc);
        switch (op) {
            .@":=" => {
                try self.lockNode(rhs);
                self.ctx.vars.items[self.ctx.vars.items.len - 1].value = rhs;
            },
            .@"=" => {
                const vr = &self.ctx.vars.items[last_var];
                self.unlockNode(vr.value);
                try self.lockNode(rhs);
                vr.value = rhs;
            },
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
            self.control = try self.alloc(.cfg_return, .{
                .cfg = self.control,
                .value = try self.nextExpr(),
            });
            return self.control;
        },
        .@"if" => {
            const cond = try self.nextExpr();
            self.control = try self.alloc(.cfg_if, .{ .cfg = self.control, .cond = cond });

            unreachable;
        },
        .Int => return try self.alloc(.const_int, .{
            .value = std.fmt.parseInt(i64, token.view(self.lexer.source), 10) catch
                unreachable,
        }),
        .Ident => {
            const view = token.view(self.lexer.source);
            if (std.mem.eql(u8, view, "arg")) {
                return self.ctx.vars.items[0].value;
            }

            for (self.ctx.vars.items, 0..) |vr, i| {
                if (std.mem.eql(u8, view, Lexer.peekStr(self.lexer.source, vr.offset))) {
                    self.last_var = @intCast(i);
                    return vr.value;
                }
            } else {
                try self.ctx.vars.append(self.gpa, .{
                    .offset = token.offset,
                    .value = undefined,
                });
            }
        },
        .@"-" => return try self.alloc(.@"uo-", .{ .oper = try self.nextUnit() }),
        .@"(" => {
            const expr = try self.nextExpr();
            std.debug.assert(self.advance().lexeme == .@")");
            return expr;
        },
        .@"{" => {
            const scope_frame = self.ctx.vars.items.len;
            while (self.cur.lexeme != .@"}") _ = try self.nextExpr();
            _ = self.advance();
            for (self.ctx.vars.items[self.ctx.vars.items.len..]) |vr| self.unlockNode(vr.value);
            self.ctx.vars.items.len = scope_frame;
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

fn alloc(self: *Parser, comptime kind: Kind, payload: kind.InputPayload()) Error!Id {
    if (try self.peephole(kind, payload)) |rpl| {
        try self.lockNode(rpl);
        defer self.unlockNode(rpl);
        const inps = Son.Inputs.idsOfPayload(&payload);
        inline for (inps) |inp| try self.lockNode(inp);
        inline for (inps) |inp| {
            self.unlockNode(inp);
            self.remove(inp);
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
                inline for (inps) |inp| {
                    self.removeNodeDep(inp, id);
                    self.remove(inp);
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
        Son.If => self.peepholeIf(kind, payload),
        else => null,
    };
}

fn peepholeBinOp(self: *Parser, comptime kind: Kind, abo: Son.BinOp) !?Id {
    var bo, var changed = .{ abo, false };
    var const_lhs = bo.lhs.kind().isConst();
    var const_rhs = bo.rhs.kind().isConst();
    const commutatuive = comptime kind.isCommutative();

    // fold
    if (const_lhs and const_rhs) return try self.peepholeAlloc(.const_int, .{
        .value = kind.applyBinOp(
            self.son.get(bo.lhs).inputs.const_int.value,
            self.son.get(bo.rhs).inputs.const_int.value,
        ),
    });

    if (bo.lhs.eql(bo.rhs) and !const_lhs) switch (kind) {
        // normalize
        .@"bo+" => return try self.alloc(.@"bo*", .{
            .lhs = bo.lhs,
            .rhs = try self.peepholeAlloc(.const_int, .{ .value = 2 }),
        }),
        // fold
        .@"bo-" => return try self.peepholeAlloc(.const_int, .{ .value = 0 }),
        // fold
        .@"bo/" => return try self.peepholeAlloc(.const_int, .{ .value = 1 }),
        else => {},
    };

    // normalize
    if (commutatuive and (bo.lhs.lessThen(bo.rhs))) {
        changed = true;
        std.mem.swap(Id, &bo.rhs, &bo.lhs);
        std.mem.swap(bool, &const_rhs, &const_lhs);
    }

    // fold
    if (commutatuive and const_rhs) {
        const rhs = self.son.get(bo.rhs).inputs.const_int.value;
        switch (kind) {
            .@"bo+", .@"bo-" => if (rhs == 0) return bo.lhs,
            .@"bo*" => if (rhs == 1)
                return bo.lhs
            else if (rhs == 0)
                return try self.peepholeAlloc(.const_int, .{ .value = 0 }),
            .@"bo/" => return bo.lhs,
            else => {},
        }
    }

    if (commutatuive and bo.lhs.kind() == kind) {
        const lhs = self.son.get(bo.lhs).inputs.bo;
        const const_lhs_rhs = lhs.rhs.kind().isConst();

        // fold
        if (const_lhs_rhs and const_rhs) return try self.alloc(kind, .{
            .lhs = lhs.lhs,
            .rhs = try self.peepholeAlloc(.const_int, .{ .value = kind.applyBinOp(
                self.son.get(lhs.rhs).inputs.const_int.value,
                self.son.get(bo.rhs).inputs.const_int.value,
            ) }),
        });

        // normalize
        if (const_lhs_rhs) return try self.alloc(kind, .{
            .lhs = try self.alloc(kind, .{ .lhs = lhs.lhs, .rhs = bo.rhs }),
            .rhs = lhs.rhs,
        });
    }

    if (kind == .@"bo*" and const_rhs and bo.lhs.kind() == .@"bo+") {
        const lhs = self.son.get(bo.lhs).inputs.bo;
        const const_lhs_rhs = lhs.rhs.kind().isConst();

        // normalize
        if (const_lhs_rhs) return try self.alloc(.@"bo+", .{
            .lhs = try self.alloc(.@"bo*", .{ .lhs = lhs.lhs, .rhs = bo.rhs }),
            .rhs = try self.peepholeAlloc(.const_int, .{
                .value = kind.applyBinOp(
                    self.son.get(lhs.rhs).inputs.const_int.value,
                    self.son.get(bo.rhs).inputs.const_int.value,
                ),
            }),
        });
    }

    // normalize
    if (kind == .@"bo-" and bo.lhs.kind() == .@"bo-") {
        const lhs = self.son.get(bo.lhs).inputs.bo;
        return try self.alloc(.@"bo-", .{
            .lhs = lhs.lhs,
            .rhs = try self.alloc(.@"bo+", .{ .lhs = lhs.rhs, .rhs = bo.rhs }),
        });
    }

    if (changed) return try self.peepholeAlloc(kind, bo);

    return null;
}

fn peepholeUnOp(self: *Parser, comptime kind: Kind, uo: Son.UnOp) !?Id {
    const const_oper = uo.oper.kind().isConst();

    // fold
    if (const_oper) return try self.peepholeAlloc(.const_int, .{
        .value = kind.applyUnOp(
            self.son.get(uo.oper).inputs.const_int.value,
        ),
    });

    return null;
}

fn peepholeIf(self: *Parser, comptime kind: Kind, f: Son.If) !?Id {
    comptime std.debug.assert(kind == .cfg_if);

    const const_cond = f.cond.kind().isConst();

    // fold
    if (const_cond) return switch (self.son.get(f.cond).inputs.const_int.value) {
        0 => try self.peepholeAlloc(.@"cfg_if:false", f),
        else => try self.peepholeAlloc(.@"cfg_if:true", f),
    };

    return null;
}

fn peepholeAlloc(self: *Parser, comptime kind: Kind, payload: kind.InputPayload()) !Id {
    const result = try self.getGwn(kind, payload);
    if (result.found_existing) return result.key_ptr.id;
    result.key_ptr.id = try self.son.add(self.gpa, kind, payload);
    inline for (self.son.get(result.key_ptr.id).inputs.idsOf(kind.inputPayloadName())) |inp| {
        try self.addNodeDep(inp, result.key_ptr.id);
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
    const entry = try parser.son.add(parser.gpa, .cfg_start, .{});
    parser.control = try parser.alloc(.cfg_tuple, .{ .on = entry, .index = 0 });

    const arg = try parser.alloc(.cfg_tuple, .{ .on = entry, .index = 1 });
    try parser.ctx.vars.append(parser.gpa, .{ .offset = Var.arg_sentinel, .value = arg });
    try parser.lockNode(arg);
    defer parser.unlockNode(arg);
    var last_node = entry;
    while (try parser.next()) |node| last_node = node;
    parser.ctx.vars.items.len = 0;
    try parser.assertIntegirty(entry);
    return .{ parser.son, .{ .entry = entry, .exit = last_node } };
}

fn lockNode(self: *Parser, id: Id) !void {
    try self.addNodeDep(id, .{});
}

fn unlockNode(self: *Parser, id: Id) void {
    self.removeNodeDep(id, .{});
}

fn addNodeDep(self: *Parser, on: Id, dep: Id) !void {
    try self.son.getPtr(on).refs.append(self.gpa, &self.son.slices, dep);
}

fn removeNodeDep(self: *Parser, on: Id, dep: Id) void {
    self.son.getPtr(on).refs.remove(&self.son.slices, dep);
}

fn constCase(exit: i64, code: []const u8) !void {
    var son, const fnc = try testParse(code);
    defer son.deinit(std.testing.allocator);
    const rvl = son.get(fnc.exit).inputs.cfg_return.value;
    std.testing.expectEqual(Kind.const_int, rvl.kind()) catch |e| {
        var output = std.ArrayList(u8).init(std.testing.allocator);
        defer output.deinit();
        try son.fmt(fnc.entry, &output);
        std.debug.print("{s}\n", .{output.items});
        return e;
    };
    const cnst = son.get(rvl).inputs.const_int.value;
    std.testing.expectEqual(exit, cnst) catch |e| {
        return e;
    };
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
    try constCase(0, "return (1 + (arg + 2)) - (arg + 3)");
    try constCase(0, "return (1 + arg + 2 + arg + 3) - (arg * 2 + 6)");
    try constCase(0, "return (arg + 0) - arg");
    try constCase(0, "return (arg * 1) - arg");
    try constCase(0, "return arg * 0");
    try constCase(0,
        \\a := arg + 2
        \\b := 1
        \\{
        \\  c := a
        \\  b = a + b
        \\}
        \\c := b
        \\return 1 + a + c - arg * 2 - 6
    );
    try constCase(0,
        \\a := 0
        \\if arg a = 1 else a = 2
        \\b := 2
        \\if arg b = 1
        \\return a - b
    );

    //var output = std.ArrayList(u8).init(std.testing.allocator);
    //defer output.deinit();
    //try dynCase("return 1 + arg + 2", &output);
    //try dynCase("return 1 + (arg + 2)", &output);
    //try resolveDynCases("with argument", output.items);
}
