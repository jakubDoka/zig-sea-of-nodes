const std = @import("std");

//const BraceFinder = struct {
//    prefix: []const u8,
//    aligned: []const Vec = &.{},
//    suffix: []const u8 = &.{},
//
//    depth: usize = 1,
//    escape: bool = false,
//
//    const vec_size = @min(std.simd.suggestVectorLength(u8) orelse 1, 64);
//    const Vec = @Vector(vec_size, u8);
//    const Mask = std.meta.Int(.unsigned, vec_size);
//
//    fn init(source: []const u8) BraceFinder {
//        if (vec_size == 1) return .{ .prefix = source };
//        const offset = std.mem.alignPointerOffset(source.ptr, @alignOf(Vec)) orelse
//            return .{ .prefix = source };
//        const simd_len = (source.len - offset) / vec_size;
//        return .{
//            .prefix = source[0..offset],
//            .aligned = @as([*]const Vec, @alignCast(@ptrCast(source.ptr + offset)))[0..simd_len],
//            .suffix = source[offset + simd_len * vec_size ..],
//        };
//    }
//
//    fn findClosing(self: *BraceFinder) ?usize {
//        if (self.findClosingUnaligned(self.prefix)) |i| return i;
//
//        const obq: Vec = @splat('{');
//        const cbq: Vec = @splat('}');
//        const esq: Vec = @splat('\\');
//
//        var carry_escape: u32 = 1;
//        for (self.aligned, 0..) |batch, i| {
//            const escapes: Mask = @bitCast(batch != esq);
//            const aligned_escapes = @shlWithOverflow(escapes, 1)[0] | carry_escape;
//            carry_escape = escapes >> vec_size - 1;
//
//            const unescaped_opening_braces: Mask = @bitCast(batch == obq);
//            const unescaped_closing_braces: Mask = @bitCast(batch == cbq);
//
//            var opening_braces = unescaped_opening_braces & aligned_escapes;
//            var closing_braces = unescaped_closing_braces & aligned_escapes;
//
//            while (opening_braces != closing_braces) {
//                self.depth -= @intFromBool(closing_braces != 0);
//                if (self.depth == 0 and @ctz(opening_braces) > @ctz(closing_braces))
//                    return i * vec_size + @ctz(closing_braces) + self.prefix.len;
//                self.depth += @intFromBool(opening_braces != 0);
//                opening_braces &= opening_braces -% 1;
//                closing_braces &= closing_braces -% 1;
//            }
//        }
//
//        if (self.findClosingUnaligned(self.suffix)) |i|
//            return i + self.prefix.len + self.aligned.len * vec_size;
//        return null;
//    }
//
//    fn findClosingUnaligned(self: *BraceFinder, src: []const u8) ?usize {
//        var i: usize = 0;
//        while (i < src.len) : (i += 1) {
//            switch (src[i]) {
//                '\\' => i += 1,
//                '{' => self.depth += 1,
//                '}' => {
//                    self.depth -= 1;
//                    if (self.depth == 0) return i;
//                },
//                else => {},
//            }
//        }
//
//        return null;
//    }
//};
//
//test {
//    const iters = 100;
//    const mult = 1;
//    const source =
//        \\
//        \\    prefix: []const u8,
//        \\    simd: []align(vec_size) const u8 = &.{},
//        \\    suffix: []const u8 = &.{},
//        \\
//        \\    depth: usize = 0,
//        \\    escape: bool = false,
//        \\
//        \\    const vec_size = std.simd.suggestVectorLength(u8);
//        \\    const Vec = @Vector(vec_size, u8);
//        \\
//        \\    fn init(source: []const u8) BraceFinder {
//        \\        const simd = std.mem.alignInSlice(source, @alignOf(Vec)) orelse
//        \\            return .{ .prefix = source }; "\}"
//        \\        return .{
//        \\            .prefix = source[0 .. @intFromPtr(simd.ptr) - @intFromPtr(source.ptr)],
//        \\            .simd = simd,
//        \\        };
//        \\    }
//        \\}
//    ** mult;
//
//    var simd_acc: u64 = 0;
//    var normal_acc: u64 = 0;
//
//    for (0..iters) |_| {
//        var now = try std.time.Timer.start();
//        var bf = BraceFinder.init(source);
//        const opos = bf.findClosingUnaligned(source);
//        normal_acc += now.lap();
//        bf = BraceFinder.init(source);
//        const pos = bf.findClosing();
//        simd_acc += now.lap();
//        std.debug.assert(pos == opos);
//    }
//
//    std.debug.print("simd   took {}ns\n", .{simd_acc / iters / mult});
//    std.debug.print("noraml took {}ns\n", .{normal_acc / iters / mult});
//}

const Lexeme = enum(u8) {
    Eof = 0,
    @"return",
    Int,
    Ident,

    @"=" = '=',
    @";" = ';',
    @"+" = '+',
    @"-" = '-',
    @"*" = '*',
    @"/" = '/',
    @"{" = '{',
    @"}" = '}',
    @"(" = '(',
    @")" = ')',

    @":=" = ':' + 128,
    @"==" = '=' + 128,
    @"!=" = '!' + 128,

    fn prec(self: Lexeme) u8 {
        return switch (self) {
            .@"=", .@":=" => 15,
            .@"==", .@"!=" => 7,
            .@"+", .@"-" => 4,
            .@"*", .@"/" => 3,
            else => 254,
        };
    }
};

const Token = packed struct(u64) {
    lexeme: Lexeme,
    length: u24,
    offset: u32,

    fn view(self: Token, source: []const u8) []const u8 {
        return source[self.offset..][0..self.length];
    }
};

fn dbg(any: anytype) @TypeOf(any) {
    std.debug.print("{any}\n", .{any});
    return any;
}

const Lexer = struct {
    source: []const u8,
    index: usize = 0,

    fn next(self: *Lexer) Token {
        while (self.index < self.source.len) {
            const start = self.index;
            self.index += 1;
            const lexeme: Lexeme = switch (self.source[start]) {
                0...32 => continue,
                'a'...'z', 'A'...'Z' => b: {
                    while (self.index < self.source.len) switch (self.source[self.index]) {
                        'a'...'z', 'A'...'Z', '0'...'9' => self.index += 1,
                        else => break,
                    };

                    inline for (std.meta.fields(Lexeme)) |field| {
                        if (comptime !std.ascii.isLower(field.name[0])) continue;
                        if (std.mem.eql(u8, field.name, self.source[start..self.index]))
                            break :b @enumFromInt(field.value);
                    }

                    break :b .Ident;
                },
                '0'...'9' => b: {
                    while (self.index < self.source.len) switch (self.source[self.index]) {
                        '0'...'9' => self.index += 1,
                        else => break,
                    };

                    break :b .Int;
                },
                ':', '=', '!' => |c| @enumFromInt(c + @as(u8, 128) * @intFromBool(self.advanceIf('='))),
                else => |c| @enumFromInt(c),
            };
            return .{
                .lexeme = lexeme,
                .length = @intCast(self.index - start),
                .offset = @intCast(start),
            };
        }

        return .{
            .lexeme = .Eof,
            .length = 0,
            .offset = 0,
        };
    }

    fn advanceIf(self: *Lexer, c: u8) bool {
        if (self.index < self.source.len and self.source[self.index] == c) {
            self.index += 1;
            return true;
        }
        return false;
    }

    fn peekStr(source: []const u8, pos: usize) []const u8 {
        var self = Lexer{ .source = source, .index = pos };
        return self.next().view(source);
    }
};

const Parser = struct {
    son: *Son,
    lexer: Lexer,
    cur: Token,
    vars: std.ArrayListUnmanaged(Variable) = .{},
    prev_cntrl: Id = undefined,

    const Variable = packed struct(u64) {
        offset: u32,
        value: Id = undefined,
    };

    const Error = error{OutOfMemory};

    fn init(lexer: Lexer, son: *Son) Parser {
        var lxr = lexer;
        return .{ .cur = lxr.next(), .lexer = lxr, .son = son };
    }

    fn deinit(self: *Parser) void {
        self.vars.deinit(self.son.gpa);
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

            const op = self.advance().lexeme;
            const rhs = try self.nextBinOp(try self.nextUnit(), next_prec);
            switch (op) {
                .@":=" => {
                    std.debug.assert(acc.tag() == .@"var");
                    self.vars.items[acc.index].value = rhs;
                    try self.son.outAppend(rhs, Id.init(.@"var", 0));
                    std.debug.assert(self.advance().lexeme == .@";");
                    return Id.init(.@"var", 0);
                },
                .@"=" => for (self.vars.items) |*variable| {
                    if (std.meta.eql(variable.value, acc)) {
                        variable.value = rhs;
                        std.debug.assert(self.advance().lexeme == .@";");
                        return Id.init(.@"var", 0);
                    }
                } else unreachable,
                else => acc = try self.alloc(switch (op) {
                    inline else => |t| if (@hasField(Node.Kind, @tagName(t)))
                        @field(Node.Kind, @tagName(t))
                    else
                        unreachable,
                }, .{ .lhs = acc, .rhs = rhs }),
            }
        }
        return acc;
    }

    fn nextUnit(self: *Parser) !Id {
        const token = self.advance();

        switch (token.lexeme) {
            .Int => return try self.alloc(.@"const", .{
                .int = std.fmt.parseInt(i64, token.view(self.lexer.source), 10) catch
                    unreachable,
            }),
            .@"-" => return try self.alloc(.@"#-", try self.nextUnit()),
            .@"return" => {
                const value = try self.nextExpr();
                std.debug.assert(self.advance().lexeme == .@";");
                const return_node = try self.alloc(.@"return", .{
                    .value = value,
                    .cfg = self.prev_cntrl,
                });
                return return_node;
            },
            .Ident => for (self.vars.items) |variable| {
                const ident_str = token.view(self.lexer.source);
                if (variable.offset == std.math.maxInt(u32) and
                    std.mem.eql(u8, "arg", ident_str))
                    return Id.init(.tuple, 2);
                const name = Lexer.peekStr(self.lexer.source, variable.offset);
                if (std.mem.eql(u8, name, ident_str))
                    return variable.value;
            } else {
                try self.vars.append(self.son.gpa, .{ .offset = @intCast(token.offset) });
                return Id.init(.@"var", self.vars.items.len - 1);
            },
            .@"{" => {
                const scope_frame = self.vars.items.len;
                while (self.cur.lexeme != .@"}") {
                    _ = try self.nextExpr();
                }
                _ = self.advance();
                for (self.vars.items[scope_frame..]) |variable| {
                    self.son.freeId(variable.value, Id.init(.@"var", 0));
                }
                self.vars.items.len = scope_frame;
                return Id.init(.@"var", 0);
            },
            .@"(" => return .{ self.nextExpr(), self.advance() }[0],
            else => |t| std.debug.panic("unimplemented lexeme: {any}", .{t}),
        }
    }

    fn alloc(self: *Parser, kind: Node.Kind, anode: anytype) !Id {
        const id = try self.son.peephole(kind, Node.init(.top, anode).inputs);
        switch (id.tag()) {
            inline else => |t| {
                const name = if (comptime std.ascii.isLower(@tagName(t)[0]))
                    @tagName(t)
                else if (comptime std.mem.startsWith(u8, @tagName(t), "#"))
                    "un_op"
                else
                    "bin_op";

                const inputs = self.son.nodes.items(.inputs)[id.index];
                const payload = @field(inputs, name);
                switch (@typeInfo(@TypeOf(payload))) {
                    .Struct => inline for (std.meta.fields(@TypeOf(payload))) |field| {
                        if (field.type == Id) {
                            try self.son.outAppend(@field(payload, field.name), id);
                        }
                    },
                    else => switch (@TypeOf(payload)) {
                        Id => try self.son.outAppend(payload, id),
                        else => {},
                    },
                }
            },
        }
        return id;
    }

    inline fn advance(self: *Parser) Token {
        defer self.cur = self.lexer.next();
        return self.cur;
    }
};

const Son = struct {
    gpa: std.mem.Allocator,
    nodes: std.MultiArrayList(Node) = .{},
    out_slices: std.ArrayListUnmanaged(Id) = .{},
    free: u32 = std.math.maxInt(u32),

    fn deinit(self: *Son) void {
        self.nodes.deinit(self.gpa);
        self.out_slices.deinit(self.gpa);
        self.* = undefined;
    }

    fn peephole(self: *Son, kind: Node.Kind, anode: Node.Inputs) !Id {
        var node = anode;
        const inputs = self.nodes.items(.inputs);
        switch (kind) {
            .@"#-" => if (self.isConst(node.un_op)) {
                defer self.freeId(node.un_op, null);
                return try self.append(.@"const", .{
                    .int = -inputs[node.un_op.index].@"const".int,
                });
            },
            inline .@"+", .@"*", .@"/", .@"-" => |op| {
                if (self.isConst(node.bin_op.lhs) and self.isConst(node.bin_op.rhs)) {
                    const lhs = inputs[node.bin_op.lhs.index].@"const".int;
                    const rhs = inputs[node.bin_op.rhs.index].@"const".int;
                    self.freeId(node.bin_op.lhs, null);
                    self.freeId(node.bin_op.rhs, null);
                    return try self.append(.@"const", .{ .int = op.apply(lhs, rhs) });
                }

                if (self.isConst(node.bin_op.lhs)) {
                    std.mem.swap(Id, &node.bin_op.lhs, &node.bin_op.rhs);
                }

                if (node.bin_op.lhs.tag().isCom()) {
                    const lhs_n = &inputs[node.bin_op.lhs.index].bin_op;

                    if (self.isConst(lhs_n.rhs) and self.isConst(node.bin_op.rhs)) {
                        const lhs = inputs[lhs_n.rhs.index].@"const".int;
                        const rhs = inputs[node.bin_op.rhs.index].@"const".int;
                        self.freeId(lhs_n.rhs, node.bin_op.lhs);
                        self.freeId(node.bin_op.rhs, null);
                        lhs_n.rhs = try self.append(.@"const", .{ .int = op.apply(lhs, rhs) });
                        return node.bin_op.lhs;
                    }

                    if (self.isConst(lhs_n.rhs)) {
                        std.mem.swap(Id, &lhs_n.rhs, &node.bin_op.rhs);
                        self.outRemove(node.bin_op.rhs, node.bin_op.lhs);
                        try self.outAppend(lhs_n.rhs, node.bin_op.lhs);
                    }
                }

                const identity: ?i64 = switch (op) {
                    .@"+" => 0,
                    .@"*" => 1,
                    else => null,
                };
                if (self.isConst(node.bin_op.rhs) and
                    inputs[node.bin_op.rhs.index].@"const".int == identity)
                    return node.bin_op.lhs;

                if (op == .@"+" and node.bin_op.lhs.index == node.bin_op.rhs.index) {
                    return try self.append(.@"*", .{
                        .lhs = node.bin_op.lhs,
                        .rhs = try self.append(.@"const", .{ .int = 2 }),
                    });
                }
            },
            else => {},
        }

        return self.append(kind, node);
    }

    fn isConst(self: *Son, id: Id) bool {
        _ = self;
        return id.tag() == .@"const";
    }

    fn append(self: *Son, kind: Node.Kind, inputs: anytype) !Id {
        if (self.free != std.math.maxInt(u32)) {
            const id = self.free;
            self.free = self.nodes.items(.out)[id].len;
            self.nodes.set(id, Node.init(.top, inputs));
            std.debug.assert(self.nodes.items(.out)[id].len == 0);
            return Id.init(kind, id);
        }
        try self.nodes.append(self.gpa, Node.init(.top, inputs));
        return Id.init(kind, self.nodes.len - 1);
    }

    fn freeId(self: *Son, id: Id, from: ?Id) void {
        if (from) |f| self.outRemove(id, f);
        if (self.nodes.items(.out)[id.index].len > 0) return;
        self.nodes.set(id.index, undefined);
        self.nodes.items(.out)[id.index].len = self.free;
        self.free = id.index;
    }

    fn outView(self: *Son, id: anytype) []Id {
        const out = switch (@TypeOf(id)) {
            *Node.Out => id,
            Id => &self.nodes.items(.out)[id.index],
            else => @compileError("wat"),
        };

        switch (out.len) {
            0 => return &[_]Id{},
            1 => return @as([*]Id, @ptrCast(&out.value.direct))[0..1],
            else => return self.out_slices.items[out.value.base..][0..id.len],
        }
    }

    fn outAppend(self: *Son, id: anytype, value: Id) !void {
        const out = switch (@TypeOf(id)) {
            *Node.Out => id,
            Id => &self.nodes.items(.out)[id.index],
            else => @compileError("wat"),
        };

        out.len += 1;
        switch (out.len) {
            1 => out.value.direct = value,
            2 => {
                const data = [_]Id{ out.value.direct, value };
                out.value = .{ .base = @intCast(self.out_slices.items.len) };
                try self.out_slices.appendSlice(self.gpa, &data);
            },
            else => try self.out_slices.append(self.gpa, value),
        }
    }

    fn outRemove(self: *Son, id: Id, from: Id) void {
        const out = &self.nodes.items(.out)[id.index];

        out.len -= 1;
        switch (out.len) {
            0 => {},
            1 => std.debug.assert(std.meta.eql(out.value.direct, from)),
            else => {
                const view = self.out_slices.items[out.value.base..][0 .. out.len + 1];
                const index = std.mem.indexOfScalar(u32, @ptrCast(view), @bitCast(from)).?;
                if (out.len == 1)
                    out.value = .{ .direct = view[index] }
                else
                    std.mem.swap(Id, &view[index], &view[out.len]);
            },
        }
    }
};

const Id = packed struct(u32) {
    tagi: std.meta.Tag(Node.Kind),
    index: std.meta.Int(.unsigned, 32 - @bitSizeOf(Node.Kind)),

    fn init(kind: Node.Kind, index: usize) Id {
        return .{ .tagi = @intFromEnum(kind), .index = @intCast(index) };
    }

    fn tag(self: Id) Node.Kind {
        return @enumFromInt(self.tagi);
    }
};

const Node = struct {
    type: Type = .top,
    out: Out = .{},
    inputs: Inputs,

    const Inputs = union {
        @"var": void,
        start: void,
        @"return": struct {
            value: Id,
            cfg: Id,
        },
        @"const": union {
            int: i64 align(4),
        },
        tuple: struct {
            on: Id,
            index: u32,
        },
        bin_op: BinOp,
        un_op: Id,
    };

    const Type = enum {
        bottom,
        top,
        IntTop,
        Int,
        IntBot,
    };

    const Kind = enum {
        @"var",
        start,
        tuple,
        @"return",
        @"const",
        @"==",
        @"!=",
        @"+",
        @"-",
        @"*",
        @"/",
        @"#-",

        fn apply(comptime op: Kind, lhs: anytype, rhs: @TypeOf(lhs)) i64 {
            return switch (op) {
                .@"+" => lhs + rhs,
                .@"*" => lhs * rhs,
                .@"/" => @divFloor(lhs, rhs),
                .@"-" => lhs - rhs,
                else => @compileError("wat"),
            };
        }

        fn isCfg(self: Kind) bool {
            switch (self) {
                .@"return", .start, .tuple => return true,
                else => return false,
            }
        }

        fn isCom(self: Kind) bool {
            switch (self) {
                .@"+", .@"*" => return true,
                else => return false,
            }
        }
    };

    const BinOp = struct {
        lhs: Id,
        rhs: Id,
    };

    const Out = struct {
        value: union { direct: Id, base: u32 } = undefined,
        len: u32 = 0,
    };

    fn InputPayload(comptime kind: Kind) type {
        return @typeInfo(Inputs).Union.fields[@intFromEnum(kind)].type;
    }

    fn init(ty: Node.Type, data: anytype) Node {
        if (@TypeOf(data) == Node) return data;
        if (@TypeOf(data) == Inputs) return .{ .inputs = data, .type = ty };
        // This somehow works and I will regret it
        const payload_name = comptime for (std.meta.fields(Inputs)) |field| {
            if (field.type == @TypeOf(data)) break field.name;
            if (@typeInfo(field.type) == .Struct) {
                var assignable: usize = 0;
                for (std.meta.fields(field.type)) |inner_field| {
                    assignable += @intFromBool(@hasField(@TypeOf(data), inner_field.name));
                }
                if (assignable == std.meta.fields(@TypeOf(data)).len) break field.name;
            } else if (@typeInfo(field.type) == .Union) {
                break for (std.meta.fields(field.type)) |inner_field| {
                    if (@hasField(@TypeOf(data), inner_field.name)) break field.name;
                } else continue;
            }
        } else @compileError("wat" ++ @typeName(@TypeOf(data)));
        return .{ .inputs = @unionInit(Inputs, payload_name, data), .type = ty };
    }

    fn typed(self: Node, ty: Type) Node {
        var node = self;
        node.type = ty;
        return node;
    }
};

const Fmt = struct {
    son: *Son,
    out: *std.ArrayList(u8),
    visited: std.ArrayListUnmanaged(Id) = .{},

    fn deinit(self: *Fmt) void {
        self.visited.deinit(self.out.allocator);
    }

    fn fmt(self: *Fmt, id: Id) !void {
        if (std.mem.indexOfScalar(u32, @ptrCast(self.visited.items), @bitCast(id)) != null)
            return;
        try self.visited.append(self.out.allocator, id);
        const inputs = self.son.nodes.items(.inputs)[id.index];

        switch (id.tag()) {
            .@"return" => {
                try self.fmt(inputs.@"return".value);
            },
            .@"+", .@"-", .@"*", .@"/", .@"==", .@"!=" => {
                try self.fmt(inputs.bin_op.lhs);
                try self.fmt(inputs.bin_op.rhs);
            },
            .@"#-" => {
                try self.fmt(inputs.un_op);
            },
            .@"var", .@"const", .tuple, .start => {},
        }

        try self.out.writer().print("n{d}: ({s}) ", .{ id.index, @tagName(id.tag()) });
        switch (id.tag()) {
            .@"const" => try self.out.writer().print("{d}", .{inputs.@"const".int}),
            .@"return" => {
                try self.out.writer().print("n{d} -> n{d}", .{
                    inputs.@"return".cfg.index,
                    inputs.@"return".value.index,
                });
            },
            .tuple => {
                try self.out.writer().print("n{d}.{d}", .{
                    inputs.tuple.on.index,
                    inputs.tuple.index,
                });
            },
            .@"var", .start => {},
            .@"#-" => {
                try self.out.writer().print("n{d}", .{inputs.un_op.index});
            },
            .@"+", .@"-", .@"*", .@"/", .@"==", .@"!=" => {
                try self.out.writer().print("n{d}, n{d}", .{
                    inputs.bin_op.lhs.index,
                    inputs.bin_op.rhs.index,
                });
            },
        }
        try self.out.append('\n');

        const outs = self.son.outView(&self.son.nodes.items(.out)[id.index]);
        switch (id.tag()) {
            .start, .tuple => for (outs) |out| if (out.tag().isCfg()) try self.fmt(out),
            .@"const", .@"return", .@"var" => {},
            .@"#-" => {},
            .@"+", .@"-", .@"*", .@"/", .@"==", .@"!=" => {},
        }
    }
};

pub fn main() !void {}

fn parse(code: []const u8, son: *Son) !Id {
    const start = try son.append(.start, {});
    var parser = Parser.init(Lexer{ .source = code }, son);
    defer parser.deinit();
    parser.prev_cntrl = try parser.alloc(.tuple, .{ .on = start, .index = 0 });
    try parser.vars.append(parser.son.gpa, .{
        .offset = std.math.maxInt(u32),
        .value = try parser.alloc(.tuple, Node.init(.IntBot, .{ .on = start, .index = 1 })),
    });
    var last_node = start;
    while (try parser.next()) |node| last_node = node;
    return last_node;
}

fn constCase(exit: i64, code: []const u8) !void {
    var son = Son{ .gpa = std.testing.allocator };
    defer son.deinit();
    const last_node = try parse(code, &son);
    const rvl = son.nodes.items(.inputs)[last_node.index].@"return".value;
    const cnst = son.nodes.items(.inputs)[rvl.index].@"const".int;
    try std.testing.expectEqual(exit, cnst);
}

fn dynCaseMany(comptime name: []const u8, cases: []const []const u8) !void {
    const gpa = std.testing.allocator;
    var output = std.ArrayList(u8).init(gpa);
    defer output.deinit();

    for (cases) |code| {
        //std.debug.print("CASE: {s}\n", .{code});
        try output.writer().print("\nCASE: {s}\n", .{code});
        var son = Son{ .gpa = gpa };
        defer son.deinit();
        _ = try parse(code, &son);
        const start = Id.init(.start, 0);
        var fmt = Fmt{ .son = &son, .out = &output };
        defer fmt.deinit();
        try fmt.fmt(start);
    }

    const old, const new = .{ "tests/" ++ name ++ ".temp.old.txt", name ++ ".temp.new.txt" };

    const update = std.process.getEnvVarOwned(gpa, "PT_UPDATE") catch "";
    defer gpa.free(update);

    if (update.len > 0) {
        try std.fs.cwd().writeFile(.{
            .sub_path = old,
            .data = std.mem.trim(u8, output.items, "\n"),
        });
    } else {
        try std.fs.cwd().writeFile(.{
            .sub_path = new,
            .data = std.mem.trim(u8, output.items, "\n"),
        });
        const err = runDiff(gpa, old, new);
        try std.fs.cwd().deleteFile(new);

        try err;
    }
}

pub fn runDiff(gpa: std.mem.Allocator, old: []const u8, new: []const u8) !void {
    var child = std.process.Child.init(&.{ "diff", "--unified", "--color", old, new }, gpa);
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

test "arithmetic" {
    try constCase(2, "return 1 + 2 * 3 + -5;");
}

test "variables-0" {
    try constCase(4,
        \\a := 1;
        \\b := 2;
        \\c := 0;
        \\{
        \\    d := 3;
        \\    c = a + d;
        \\}
        \\return c;
    );
}

test "variables-1" {
    try constCase(8,
        \\x0 := 1;
        \\y0 := 2;
        \\x1 := 3;
        \\y1 := 4;
        \\return (x0 - x1)*(x0 - x1) + (y0 - y1)*(y0 - y1);
    );
}

test "unnown-arguments" {
    try dynCaseMany("unnown-arguments-0", &.{
        "return 1 + arg + 2;",
        "return (1 + arg) + 2;",
        "return 0 + arg;",
        "return arg + 0 + arg;",
        "return 1 + arg + 2 + arg + 3;",
        "return 1 * arg;",
        "return 3 == 3;",
        "return 3 == 4;",
        "return 3 != 3;",
        "return 3 != 4;",
        "a := arg+1; b := a; b = 1; return a + 2;",
        "a := arg + 1; a = a; return a;",
        "return -arg;",
    });
}
