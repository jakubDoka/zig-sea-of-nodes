lexer: Lexer,
cur: Token = undefined,
son: Son = .{},
control: Id = undefined,
end: Id = undefined,
gpa: std.mem.Allocator,
last_var: u31 = undefined,
gvn: gvn.Map = .{},
vars: std.MultiArrayList(Var) = .{},
ctx: struct {
    vdups: std.ArrayListUnmanaged(Id) = .{},
    loops: std.ArrayListUnmanaged(Loop) = .{},
    worklist: std.ArrayListUnmanaged(Id) = .{},
    // maybe we can getaway with fixed buffer
    peep_deps: std.ArrayListUnmanaged(Id) = .{},
} = .{},

const std = @import("std");
const main = @import("main.zig");
const debug = @import("builtin").mode == .Debug;
const Lexer = @import("Lexer.zig");
const Token = Lexer.Token;
const Son = @import("Son.zig");
const Kind = Son.Kind;
const Fn = Son.Fn;
const Node = Son.Node;
const Id = Son.Id;
const Interpreter = @import("Interpreter.zig");
const Parser = @This();
const Error = std.mem.Allocator.Error;

fn swapSslices(comptime T: type, a: []T, b: []T) void {
    for (a, b) |*ae, *be| std.mem.swap(T, ae, be);
}

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

const VarChange = struct {
    var_index: packed struct { right: bool = true, i: u31 },
    prev_value: Id,

    fn lessThen(_: void, a: VarChange, b: VarChange) bool {
        return @as(u32, @bitCast(a.var_index)) < @as(u32, @bitCast(b.var_index));
    }

    fn cmp(a: u31, b: VarChange) std.math.Order {
        return std.math.order(a, b.var_index.i);
    }
};

const Loop = struct {
    entry: Id,
    exit: Id = .{},
    @"continue": Id = .{},
    var_base: usize,
    cached_var_base: usize,
};

pub fn assertIntegirty(self: *Parser, entry: Id) !void {
    var leaked = std.ArrayList(Id).init(self.gpa);
    defer leaked.deinit();
    try self.son.collectLeakedIds(entry, &leaked);
    for (leaked.items) |it| self.son.logNode(it);
    std.testing.expectEqualSlices(Id, &.{}, leaked.items) catch |e| {
        try self.son.log(entry, self.gpa);
        return e;
    };
}

pub fn deinit(self: *Parser) void {
    inline for (@typeInfo(@TypeOf(self.ctx)).Struct.fields) |field| {
        switch (field.type) {
            usize => if (@field(self.ctx, field.name) != 0) std.debug.panic("{s}", .{field.name}),
            else => {
                if (@field(self.ctx, field.name).items.len != 0) std.debug.panic("{s}", .{field.name});
                @field(self.ctx, field.name).deinit(self.gpa);
            },
        }
    }
    std.debug.assert(self.vars.len == 0);
    self.vars.deinit(self.gpa);
    self.gvn.deinit(self.gpa);
    self.* = undefined;
}

pub fn next(self: *Parser) !?Id {
    if (self.cur.lexeme == .Eof) return null;
    return try self.nextExpr();
}

pub fn iterPeepholes(self: *Parser, iters: usize) !void {
    var list = self.ctx.worklist.toManaged(self.gpa);
    defer {
        list.items.len = 0;
        self.ctx.worklist = list.moveToUnmanaged();
    }
    try self.son.collectIds(&list);
    for (list.items, 0..) |vr, i| {
        self.son.getPtr(vr).peep_pos = @intCast(i);
    }
    for (0..iters) |_| {
        const target = list.popOrNull() orelse break;
        const new = try self.latePeephole(target);
        if (!new.eql(target)) {
            const nd = self.son.getPtr(new);
            nd.peep_pos = std.math.maxInt(u32);
            const base = list.items.len;
            try list.appendSlice(nd.refs.view(self.son.slices));
            try list.appendSlice(nd.peep_deps.view(self.son.slices));
            nd.peep_deps.clear(&self.son.slices);

            var writer = base;
            for (list.items[base..]) |vr| {
                if (self.son.get(vr).peep_pos != std.math.maxInt(u32)) continue;
                self.son.getPtr(vr).peep_pos = @intCast(writer);
                list.items[writer] = vr;
                writer += 1;
            }
            list.items.len = writer;
        }
    }
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
                self.vars.items(.value)[self.vars.len - 1] = rhs;
            },
            .@"=" => {
                const vr = &self.vars.items(.value)[last_var];
                self.unlockAndRemove(vr.*);
                try self.lockNode(rhs);
                vr.* = rhs;
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
            _ = try self.alloc(.cfg_return, .{
                .cfg = self.control,
                .end = self.end,
                .value = try self.nextExpr(),
            });
            self.control = .{};
        },
        .@"if" => {
            const prev_control = self.control;
            try self.lockNode(prev_control);
            defer self.unlockNode(prev_control);

            const cond = try self.nextExpr();
            const if_control = try self.alloc(.cfg_if, .{ .cfg = self.control, .cond = cond });

            const var_cache_base = self.ctx.vdups.items.len;
            defer self.ctx.vdups.items.len = var_cache_base;
            @memcpy(try self.ctx.vdups.addManyAsSlice(self.gpa, self.vars.len), self.vars.items(.value));
            for (self.vars.items(.value)) |vr| try self.lockNode(vr);

            {
                try self.lockNode(if_control);
                defer self.unlockNode(if_control);
                self.control = try self.alloc(.cfg_tuple, .{ .cfg = if_control, .index = 0 });
            }
            _ = try self.nextExpr();
            const then = self.control;

            swapSslices(Id, self.vars.items(.value), self.ctx.vdups.items[var_cache_base..]);

            self.control = try self.alloc(.cfg_tuple, .{ .cfg = if_control, .index = 1 });
            if (self.cur.lexeme == .@"else") {
                _ = self.advance();
                _ = try self.nextExpr();
            }
            const @"else" = self.control;

            self.control = try self.allocNopi(.cfg_region, .{ .lcfg = then, .rcfg = @"else" });
            try self.lockNode(self.control);

            for (self.vars.items(.value), self.ctx.vdups.items[var_cache_base..]) |*vr, cch| {
                self.unlockNode(cch);
                if (vr.eql(cch)) continue;
                const phi = try self.alloc(.phi, .{
                    .region = self.control,
                    .left = cch,
                    .right = vr.*,
                });
                try self.lockNode(phi);
                self.unlockNode(vr.*);
                vr.* = phi;
            }

            self.unlockNode(self.control);
            self.control = try self.latePeephole(self.control);
        },
        .loop => {
            const var_cache_base = self.ctx.vdups.items.len;
            defer self.ctx.vdups.items.len = var_cache_base;
            @memcpy(try self.ctx.vdups.addManyAsSlice(self.gpa, self.vars.len), self.vars.items(.value));
            @memset(self.vars.items(.value), .{});

            // break dup
            @memset(try self.ctx.vdups.addManyAsSlice(self.gpa, self.vars.len), .{});
            // continue dup
            @memset(try self.ctx.vdups.addManyAsSlice(self.gpa, self.vars.len), .{});

            var loop = try self.allocNopi(.@"cfg_region:loop", .{
                .lcfg = self.control,
                .rcfg = .{},
            });
            try self.ctx.loops.append(self.gpa, .{
                .entry = loop,
                .var_base = self.vars.len,
                .cached_var_base = var_cache_base,
            });

            self.control = loop;
            _ = try self.nextExpr();

            const loop_data = self.ctx.loops.pop();
            if (!loop_data.@"continue".eql(.{})) {
                const dups = self.ctx.vdups.items[var_cache_base + self.vars.len * 2 ..][0..self.vars.len];
                self.control = try self.jumpTo(self.vars.items(.value), dups, loop_data.@"continue", self.control);
                @memcpy(self.vars.items(.value), dups);
            }

            loop = try self.modifyInputs(loop, "cfg_region", "rcfg", self.control);
            try self.lockNode(loop);
            defer self.unlockNode(loop);

            try self.lockNode(loop_data.exit);
            defer self.unlockNode(loop_data.exit);

            for (
                self.vars.items(.value),
                self.ctx.vdups.items[var_cache_base..][0..self.vars.len],
                self.ctx.vdups.items[var_cache_base + self.vars.len ..][0..self.vars.len],
            ) |*vr, cch, *bcch| {
                if (!vr.eql(.{})) {
                    const phi = self.son.getPtr(cch);
                    self.unlockNode(vr.*);
                    self.unlockNode(cch);
                    if (cch.eql(vr.*)) {
                        try self.lockNode(phi.inputs.phi.left);
                        try self.patchReferences(cch, phi.inputs.phi.left);
                        vr.* = phi.inputs.phi.left;
                    } else {
                        vr.* = try self.modifyInputs(cch, "phi", "right", vr.*);
                        try self.lockNode(vr.*);
                    }
                } else {
                    vr.* = cch;
                }

                if (!bcch.eql(.{})) {
                    if (bcch.kind() == .phi) {
                        const nd = self.son.get(bcch.*);
                        std.debug.assert(!nd.inputs.phi.right.eql(.{}));
                        if (nd.inputs.phi.left.eql(.{})) {
                            std.debug.assert(!vr.eql(.{}));
                            self.unlockNode(bcch.*);
                            bcch.* = try self.modifyInputs(bcch.*, "phi", "left", vr.*);
                            bcch.* = try self.latePeephole(bcch.*);
                            try self.lockNode(bcch.*);
                        }
                    }
                    if (!vr.eql(.{})) self.unlockNode(vr.*);
                    vr.* = bcch.*;
                }
            }

            if (debug) for (self.vars.items(.value)) |vr| std.debug.assert(!vr.eql(.{}));
            //if (!loop_data.exit.eql(.{})) {
            //for (self.ctx.vdups.items[var_cache_base + self.vars.len ..][0..self.vars.len]) |vr| {
            //    if (!vr.eql(.{})) self.unlockAndRemove(vr);
            //}
            //}

            self.control = loop_data.exit;
        },
        .@"continue" => {
            const loop = &self.ctx.loops.items[self.ctx.loops.items.len - 1];
            loop.@"continue" = try self.jumpTo(
                self.vars.items(.value)[0..loop.var_base],
                self.ctx.vdups.items[loop.cached_var_base + loop.var_base * 2 ..][0..loop.var_base],
                loop.@"continue",
                self.control,
            );
            self.control = .{};
        },
        .@"break" => {
            const loop = &self.ctx.loops.items[self.ctx.loops.items.len - 1];
            loop.exit = try self.jumpTo(
                self.vars.items(.value)[0..loop.var_base],
                self.ctx.vdups.items[loop.cached_var_base + loop.var_base ..][0..loop.var_base],
                loop.exit,
                self.control,
            );
            self.control = .{};
        },
        .Int => return try self.allocNopi(.const_int, .{
            .value = std.fmt.parseInt(i64, token.view(self.lexer.source), 10) catch
                unreachable,
        }),
        .true => return try self.allocNopi(.const_int, .{ .value = 1 }),
        .false => return try self.allocNopi(.const_int, .{ .value = 0 }),
        .Ident => e: {
            const view = token.view(self.lexer.source);
            b: {
                if (std.mem.eql(u8, view, "arg")) {
                    self.last_var = 0;
                    break :b;
                }

                for (self.vars.items(.offset), 0..) |vr, i| {
                    if (std.mem.eql(u8, view, Lexer.peekStr(self.lexer.source, vr))) {
                        self.last_var = @intCast(i);
                        break :b;
                    }
                } else {
                    try self.vars.append(self.gpa, .{
                        .offset = token.offset,
                        .value = undefined,
                    });
                    break :e;
                }
            }

            const vr = &self.vars.items(.value)[self.last_var];
            if (vr.eql(.{})) {
                const loop = self.ctx.loops.getLast();
                const orig = &self.ctx.vdups.items[loop.cached_var_base + self.last_var];
                const phi = try self.allocNopi(.phi, .{
                    .region = loop.entry,
                    .left = orig.*,
                    .right = .{},
                });
                self.unlockNode(orig.*);
                try self.lockNode(phi);
                try self.lockNode(phi);
                orig.* = phi;
                vr.* = phi;
            }

            return vr.*;
        },
        .@"-" => return try self.alloc(.@"uo-", .{ .oper = try self.nextUnit() }),
        .@"(" => {
            const expr = try self.nextExpr();
            std.debug.assert(self.advance().lexeme == .@")");
            return expr;
        },
        .@"{" => {
            const scope_frame = self.vars.len;
            while (self.cur.lexeme != .@"}") _ = try self.nextExpr();
            _ = self.advance();
            for (self.vars.items(.value)[self.vars.len..]) |vr| self.unlockAndRemove(vr);
            self.vars.len = scope_frame;
        },
        else => |e| std.debug.panic("unhandled token: {s}", .{@tagName(e)}),
    }
    return .{};
}

fn jumpTo(self: *Parser, vars: []Id, dups: []Id, target: Id, from: Id) !Id {
    if (target.eql(.{})) {
        for (vars, dups) |*vr, *cch| {
            if (cch.eql(vr.*)) continue;
            try self.lockNode(vr.*);
            cch.* = vr.*;
        }
        return from;
    } else {
        const region = try self.allocNopi(.cfg_region, .{ .lcfg = target, .rcfg = from });
        try self.lockNode(region);
        defer self.unlockNode(region);

        for (vars, dups) |*vr, *cch| {
            if (cch.eql(vr.*)) continue;
            const phi = try self.alloc(.phi, .{
                .region = region,
                .left = cch.*,
                .right = vr.*,
            });
            try self.lockNode(phi);
            if (!cch.eql(.{})) self.unlockNode(cch.*);
            cch.* = phi;
        }
        return try self.latePeephole(region);
    }
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
        for (self.ctx.peep_deps.items) |id| {
            try self.son.getPtr(id).peep_deps.append(self.gpa, &self.son.slices, rpl);
        }
        self.ctx.peep_deps.items.len = 0;

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

    return try self.allocNopi(kind, payload);
}

fn latePeephole(self: *Parser, id: Id) !Id {
    const rpl = switch (id.kind()) {
        inline else => |t| try self.peephole(t, @field(self.son.get(id).inputs, t.inputPayloadName())),
    } orelse {
        for (self.ctx.peep_deps.items) |tid| {
            try self.son.getPtr(tid).peep_deps.append(self.gpa, &self.son.slices, id);
        }
        self.ctx.peep_deps.items.len = 0;
        return id;
    };

    for (self.ctx.peep_deps.items) |tid| {
        try self.son.getPtr(tid).peep_deps.append(self.gpa, &self.son.slices, rpl);
    }
    self.ctx.peep_deps.items.len = 0;

    try self.lockNode(rpl);
    defer self.unlockNode(rpl);
    try self.patchReferences(id, rpl);
    return rpl;
}

fn patchReferences(self: *Parser, id: Id, rpl: Id) !void {
    for (self.son.getPtr(id).inputs.idsOfAny(id.kind())) |did| {
        for (self.son.getPtr(did).refs.view(self.son.slices)) |*v| {
            if (v.eql(id)) v.* = rpl;
        }
    }

    for (self.son.getPtr(id).refs.view(self.son.slices)) |did| {
        switch (did.kind()) {
            inline else => |t| {
                const nd = self.son.getPtr(did);
                const inputs = &@field(nd.inputs, t.inputPayloadName());
                var notFound = true;
                inline for (@typeInfo(@TypeOf(inputs.*)).Struct.fields) |field| {
                    if (field.type != Id) continue;
                    const inp = &@field(inputs, field.name);
                    if (inp.eql(id)) {
                        const node = try self.modifyInputs(did, t.inputPayloadName(), field.name, rpl);
                        try self.addNodeDep(rpl, node);
                        notFound = false;
                    }
                }
                if (notFound) return error.OutOfMemory;
            },
        }
    }
    self.son.getPtr(id).refs.clear(&self.son.slices);
    self.son.rmeove(id);
}

fn unlockAndRemove(self: *Parser, id: Id) void {
    self.unlockNode(id);
    self.remove(id);
}

fn remove(self: *Parser, id: Id) void {
    const nd = self.son.getPtr(id);
    if (nd.refs.len() == 0) {
        std.debug.assert(self.gvn.remove(gvn.Key{
            .id = id,
            .hash = gvn.hashInput(id.kind(), self.son.get(id).inputs),
        }));
        switch (id.kind()) {
            inline else => |t| {
                const inps = nd.inputs.idsOf(t.inputPayloadName());
                inline for (inps) |inp| {
                    self.removeNodeDep(inp, id);
                    self.remove(inp);
                }
            },
        }
        self.son.rmeove(id);
    }
}

fn modifyInputs(self: *Parser, on: Id, comptime payload: []const u8, comptime field: []const u8, to: Id) Error!Id {
    const payl = &@field(self.son.getPtr(on).inputs, payload);
    const input: *Id = &@field(payl, field);
    std.debug.assert(!input.eql(to));
    std.debug.assert(self.gvn.remove(gvn.Key{ .id = on, .hash = gvn.hash(on.kind(), payl.*) }));
    const prev = input.*;
    input.* = to;
    const entry = try self.getGwn(on.kind(), payl.*);
    if (!entry.found_existing) {
        self.removeNodeDep(prev, on);
        try self.addNodeDep(to, on);
        entry.key_ptr.id = on;
        return on;
    }
    input.* = prev;
    try self.patchReferences(on, entry.key_ptr.id);
    self.remove(on);
    return entry.key_ptr.id;
}

fn getGwn(self: *Parser, kind: Kind, payload: anytype) !gvn.Map.GetOrPutResult {
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
    std.debug.assert(self.ctx.peep_deps.items.len == 0);

    if (comptime for (@typeInfo(@TypeOf(payload)).Struct.fields) |field| {
        if (std.mem.endsWith(u8, field.name, "cfg")) break true;
    } else false) {
        var is_unreachable = true;
        var last_node: Id = undefined;
        inline for (@typeInfo(@TypeOf(payload)).Struct.fields) |field| {
            if (comptime !std.mem.endsWith(u8, field.name, "cfg")) continue;
            last_node = @field(payload, field.name);
            is_unreachable = is_unreachable and last_node.kind().isUnreachable();
        }

        if (is_unreachable) return last_node;
    }

    return switch (@TypeOf(payload)) {
        Son.BinOp => self.peepholeBinOp(kind, payload),
        Son.UnOp => self.peepholeUnOp(kind, payload),
        Son.If => self.peepholeIf(kind, payload),
        Son.Tuple => self.peepholeTuple(kind, payload),
        Son.Phi => self.peepholePhi(kind, payload),
        Son.Region => self.peepholeRegion(kind, payload),
        else => null,
    };
}

fn peepholeBinOp(self: *Parser, comptime kind: Kind, abo: Son.BinOp) !?Id {
    var bo, var changed = .{ abo, false };
    var const_lhs = bo.lhs.kind().isConst();
    var const_rhs = bo.rhs.kind().isConst();
    const commutatuive = comptime kind.isCommutative();

    // fold
    if (const_lhs and const_rhs) return try self.allocNopi(.const_int, .{
        .value = kind.applyBinOp(
            self.son.get(bo.lhs).inputs.const_int.value,
            self.son.get(bo.rhs).inputs.const_int.value,
        ),
    });

    if (bo.lhs.eql(bo.rhs) and !const_lhs) switch (kind) {
        // normalize
        .@"bo+" => return try self.alloc(.@"bo*", .{
            .lhs = bo.lhs,
            .rhs = try self.allocNopi(.const_int, .{ .value = 2 }),
        }),
        // fold
        .@"bo-" => return try self.allocNopi(.const_int, .{ .value = 0 }),
        // fold
        .@"bo/" => return try self.allocNopi(.const_int, .{ .value = 1 }),
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
                return try self.allocNopi(.const_int, .{ .value = 0 }),
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
            .rhs = try self.allocNopi(.const_int, .{ .value = kind.applyBinOp(
                self.son.get(lhs.rhs).inputs.const_int.value,
                self.son.get(bo.rhs).inputs.const_int.value,
            ) }),
        });

        // normalize
        if (const_lhs_rhs) return try self.alloc(kind, .{
            .lhs = try self.alloc(kind, .{ .lhs = lhs.lhs, .rhs = bo.rhs }),
            .rhs = lhs.rhs,
        });

        try self.ctx.peep_deps.append(self.gpa, lhs.rhs);
    }

    if (kind == .@"bo*" and const_rhs and bo.lhs.kind() == .@"bo+") {
        const lhs = self.son.get(bo.lhs).inputs.bo;
        const const_lhs_rhs = lhs.rhs.kind().isConst();

        // normalize
        if (const_lhs_rhs) return try self.alloc(.@"bo+", .{
            .lhs = try self.alloc(.@"bo*", .{ .lhs = lhs.lhs, .rhs = bo.rhs }),
            .rhs = try self.allocNopi(.const_int, .{
                .value = kind.applyBinOp(
                    self.son.get(lhs.rhs).inputs.const_int.value,
                    self.son.get(bo.rhs).inputs.const_int.value,
                ),
            }),
        });

        try self.ctx.peep_deps.append(self.gpa, lhs.rhs);
    }

    // normalize
    if (kind == .@"bo-" and bo.lhs.kind() == .@"bo-") {
        const lhs = self.son.get(bo.lhs).inputs.bo;
        return try self.alloc(.@"bo-", .{
            .lhs = lhs.lhs,
            .rhs = try self.alloc(.@"bo+", .{ .lhs = lhs.rhs, .rhs = bo.rhs }),
        });
    }

    if (changed) return try self.allocNopi(kind, bo);

    return null;
}

fn peepholeUnOp(self: *Parser, comptime kind: Kind, uo: Son.UnOp) !?Id {
    const const_oper = uo.oper.kind().isConst();

    // fold
    if (const_oper) return try self.allocNopi(.const_int, .{
        .value = kind.applyUnOp(
            self.son.get(uo.oper).inputs.const_int.value,
        ),
    });

    return null;
}

fn peepholeIf(self: *Parser, comptime kind: Kind, f: Son.If) !?Id {
    if (kind != .cfg_if) return null;

    const const_cond = f.cond.kind().isConst();

    // fold
    if (const_cond) return switch (self.son.get(f.cond).inputs.const_int.value) {
        0 => try self.allocNopi(.@"cfg_if:false", f),
        else => try self.allocNopi(.@"cfg_if:true", f),
    };

    var cursor: ?Id = f.cfg;
    while (cursor) |nxt| {
        if (nxt.kind() == .cfg_tuple) {
            const tup = self.son.get(nxt).inputs.cfg_tuple;
            if (tup.cfg.kind() == .cfg_if and
                self.son.get(tup.cfg).inputs.cfg_if.cond.eql(f.cond))
                return switch (tup.index) {
                    0 => try self.allocNopi(.@"cfg_if:true", f),
                    1 => try self.allocNopi(.@"cfg_if:true", f),
                    else => unreachable,
                };
        }
        cursor = self.dominatorOf(nxt);
    }

    return null;
}

fn dominatorOf(self: *Parser, id: Id) ?Id {
    const node = self.son.get(id).inputs;
    switch (id.kind()) {
        .cfg_region => {
            var lcfg = node.cfg_region.lcfg;
            var rcfg = node.cfg_region.rcfg;
            while (!lcfg.eql(rcfg)) {
                const ldepth = self.son.get(lcfg).refs.meta.depth;
                const rdepth = self.son.get(rcfg).refs.meta.depth;
                std.debug.assert(ldepth != 0);
                std.debug.assert(rdepth != 0);
                if (ldepth >= rdepth) lcfg = self.dominatorOf(lcfg) orelse return null;
                if (rdepth >= rdepth) rcfg = self.dominatorOf(rcfg) orelse return null;
            }
            return lcfg;
        },
        inline else => |t| {
            const payload = @field(node, t.inputPayloadName());
            if (!@hasField(@TypeOf(payload), "cfg")) return null;
            return payload.cfg;
        },
    }
}

fn peepholeTuple(self: *Parser, comptime kind: Kind, tuple: Son.Tuple) !?Id {
    comptime std.debug.assert(kind == .cfg_tuple);

    if (tuple.cfg.kind() == .@"cfg_if:true") return switch (tuple.index) {
        0 => self.son.get(tuple.cfg).inputs.cfg_if.cfg,
        1 => .{},
        else => unreachable,
    };

    if (tuple.cfg.kind() == .@"cfg_if:false") return switch (tuple.index) {
        0 => .{},
        1 => self.son.get(tuple.cfg).inputs.cfg_if.cfg,
        else => unreachable,
    };

    return null;
}

fn peepholeRegion(_: *Parser, comptime kind: Kind, region: Son.Region) !?Id {
    if (kind == .@"cfg_region:loop") return null;
    comptime std.debug.assert(kind == .cfg_region);

    if (region.lcfg.kind().isUnreachable()) {
        std.debug.assert(!region.rcfg.kind().isUnreachable());
        return region.rcfg;
    }
    if (region.rcfg.kind().isUnreachable()) {
        std.debug.assert(!region.lcfg.kind().isUnreachable());
        return region.lcfg;
    }

    return null;
}

fn peepholePhi(self: *Parser, comptime kind: Kind, phi: Son.Phi) !?Id {
    comptime std.debug.assert(kind == .phi);

    // fold
    if (phi.left.eql(phi.right)) return phi.left;

    const region = self.son.get(phi.region).inputs.cfg_region;
    // fold
    if (region.lcfg.kind().isUnreachable()) return phi.right;
    // fold
    if (region.rcfg.kind().isUnreachable()) return phi.left;

    // normalize
    if (phi.left.kind() == phi.right.kind() and phi.left.kind().isBinOp()) {
        const left = self.son.get(phi.left).inputs.bo;
        const right = self.son.get(phi.left).inputs.bo;
        return try self.allocNopiAny(phi.left.kind(), .{ .bo = .{
            .lhs = try self.alloc(.phi, .{ .region = phi.region, .left = left.lhs, .right = right.lhs }),
            .rhs = try self.alloc(.phi, .{ .region = phi.region, .left = left.rhs, .right = right.rhs }),
        } });
    }

    return null;
}

fn allocNopiAny(self: *Parser, kind: Kind, payload: Son.Inputs) !Id {
    return switch (kind) {
        inline else => |t| self.allocNopi(t, @field(payload, t.inputPayloadName())),
    };
}

fn allocNopi(self: *Parser, comptime kind: Kind, payload: kind.InputPayload()) !Id {
    const result = try self.getGwn(kind, payload);
    if (!result.found_existing) {
        result.key_ptr.id = try self.son.add(self.gpa, kind, payload);
        inline for (self.son.getPtr(result.key_ptr.id).inputs.idsOf(kind.inputPayloadName())) |inp| {
            try self.addNodeDep(inp, result.key_ptr.id);
        }
        if (@hasField(@TypeOf(payload), "cfg")) {
            self.son.getPtr(result.key_ptr.id).refs.meta.depth =
                self.son.get(payload.cfg).refs.meta.depth + 1;
        } else if (kind == .cfg_region) {
            self.son.getPtr(result.key_ptr.id).refs.meta.depth = @max(
                self.son.get(payload.lcfg).refs.meta.depth,
                self.son.get(payload.rcfg).refs.meta.depth,
            ) + 1;
        }
    }

    for (self.ctx.peep_deps.items) |id| {
        try self.son.getPtr(id).peep_deps.append(self.gpa, &self.son.slices, result.key_ptr.id);
    }
    self.ctx.peep_deps.items.len = 0;

    return result.key_ptr.id;
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

fn testParse(code: []const u8) !struct { Son, Fn } {
    var parser = Parser{
        .gpa = std.testing.allocator,
        .lexer = Lexer{ .source = code },
    };
    defer parser.deinit();
    errdefer parser.son.deinit(std.testing.allocator);
    parser.cur = parser.lexer.next();
    const entry = try parser.allocNopi(.cfg_start, .{});
    parser.control = try parser.allocNopi(.cfg_tuple, .{ .cfg = entry, .index = 0 });
    parser.end = try parser.allocNopi(.cfg_end, .{});
    try parser.lockNode(parser.end);

    const arg = try parser.allocNopi(.cfg_tuple, .{ .cfg = entry, .index = 1 });
    try parser.vars.append(parser.gpa, .{ .offset = Var.arg_sentinel, .value = arg });
    try parser.lockNode(arg);
    while (try parser.next() != null) {}
    for (parser.vars.items(.value)) |vr| {
        parser.unlockNode(vr);
    }
    parser.unlockNode(parser.end);
    parser.vars.len = 0;

    try parser.assertIntegirty(entry);
    try parser.son.log(entry, parser.gpa);
    try parser.iterPeepholes(0);
    try parser.assertIntegirty(entry);
    return .{ parser.son, .{ .entry = entry, .exit = parser.end } };
}

fn constCase(exit: i64, code: []const u8) !void {
    var son, const fnc = try testParse(code);
    defer son.deinit(std.testing.allocator);
    for (son.getPtr(fnc.exit).refs.view(son.slices)) |ret| {
        const rvl = son.get(ret).inputs.cfg_return.value;
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
}

fn runCase(code: []const u8, comptime mock: fn (i64) i64, _: *std.Random.DefaultPrng) !void {
    var son, const fnc = try testParse(code);
    defer son.deinit(std.testing.allocator);

    //try son.log(fnc.entry, std.testing.allocator);

    var inter = try Interpreter.init(&son, std.testing.allocator);
    defer inter.deinit(std.testing.allocator);

    var failed = false;
    for (0..10) |i| {
        const arg: u32 = @intCast(i);
        const mock_res = mock(arg);
        inter.computed[3] = arg;
        const inter_res = inter.run(fnc.entry) catch |e| {
            std.debug.print("interpreter failed on arg: {d} {s}\n", .{ arg, @errorName(e) });
            failed = true;
            continue;
        };
        std.testing.expectEqual(mock_res, inter_res) catch {
            std.debug.print("interpreter mismatched on arg: {d}\n", .{arg});
            failed = true;
            continue;
        };
    }
    if (failed) try son.log(fnc.entry, std.testing.allocator);
    try std.testing.expect(!failed);
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
        \\b := 2
        \\if arg {
        \\  a = 1
        \\  b = 1
        \\} else a = 2
        \\return a - b
    );
    try constCase(0,
        \\if true return 0
        \\return 2
    );
    try constCase(0,
        \\if false return 2
        \\return 0
    );
    try constCase(0,
        \\if arg {
        \\  if arg - 1 {}
        \\  if arg return 0
        \\  return 1
        \\}
        \\return 0
    );

    const hblang_impls = .{
        \\a := 1
        \\loop if arg < 10 arg = arg + a else break
        \\return arg
        ,
        \\loop if arg < 10 {
        \\  arg = arg + 1
        \\  if arg == 3 break
        \\} else break
        \\return arg
        ,
        \\loop if arg < 10 {
        \\  if arg == 3 {
        \\      arg = 0
        \\      break
        \\  }
        \\  arg = arg + 1
        \\} else break
        \\return arg
        ,
        \\loop if arg < 10 {
        \\  if arg == 2 {
        \\      arg = 4
        \\      continue
        \\  }
        \\  if arg == 3 {
        \\      arg = 0
        \\      break
        \\  }
        \\  arg = arg + 1
        \\} else break
        \\return arg
        ,
        \\loop if arg < 10 {
        \\  arg = arg + 1
        \\  if arg == 5 continue
        \\  if arg == 6 break
        \\} else break
        \\return arg
        ,
        \\loop if arg < 10 {
        \\  arg = arg + 1
        \\  if arg == 5 continue
        \\  if arg == 6 continue
        \\} else break
        \\return arg
        ,
        \\loop if arg < 10 {
        \\  arg = arg + 1
        \\  if arg == 5 break
        \\  if arg == 6 break
        \\} else break
        \\return arg
        ,
        \\step := 1
        \\loop if arg < 10 {
        \\    arg = arg + step + 1
        \\} else break
        \\return arg
    };
    const zig_impls = struct {
        pub fn basic_while(argv: i64) i64 {
            var arg = argv;
            while (arg < 10) {
                arg += 1;
            }
            return arg;
        }
        pub fn two_breaks(argv: i64) i64 {
            var arg = argv;
            while (arg < 10) {
                arg += 1;
                if (arg == 3) break;
            }
            return arg;
        }
        pub fn two_breaks_post(argv: i64) i64 {
            var arg = argv;
            while (arg < 10) {
                if (arg == 3) {
                    arg = 0;
                    break;
                }
                arg += 1;
            }
            return arg;
        }
        pub fn unreachable_break(argv: i64) i64 {
            var arg = argv;
            while (arg < 10) {
                if (arg == 2) {
                    arg = 4;
                    continue;
                }
                if (arg == 3) {
                    arg = 0;
                    break;
                }
                arg += 1;
            }
            return arg;
        }
        pub fn example1(argv: i64) i64 {
            var arg = argv;
            while (arg < 10) {
                arg = arg + 1;
                if (arg == 5) continue;
                if (arg == 6) break;
            }
            return arg;
        }
        pub fn example2(argv: i64) i64 {
            var arg = argv;
            while (arg < 10) {
                arg = arg + 1;
                if (arg == 5) continue;
                if (arg == 6) continue;
            }
            return arg;
        }
        pub fn example3(argv: i64) i64 {
            var arg = argv;
            while (arg < 10) {
                arg = arg + 1;
                if (arg == 5) break;
                if (arg == 6) break;
            }
            return arg;
        }
        pub fn iter_peephole(argv: i64) i64 {
            var arg = argv;
            const step = 1;
            while (arg < 10) {
                arg = arg + step + 1;
            }
            return arg;
        }
    };

    var failed = false;
    var rng = std.Random.DefaultPrng.init(0);
    inline for (hblang_impls, @typeInfo(zig_impls).Struct.decls) |hi, zi| {
        runCase(hi, @field(zig_impls, zi.name), &rng) catch |e| {
            std.debug.print("CASE({s}): {s} {s}\n", .{ zi.name, @errorName(e), hi });
            failed = true;
        };
    }
    try std.testing.expect(!failed);

    //var output = std.ArrayList(u8).init(std.testing.allocator);
    //defer output.deinit();
    //try dynCase("return 1 + arg + 2", &output);
    //try dynCase("return 1 + (arg + 2)", &output);
    //try resolveDynCases("with argument", output.items);
}
