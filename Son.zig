nodes: []Slot = &.{},
free: u32 = sentinel,
slices: Slices = .{},

const std = @import("std");
const debug = @import("builtin").mode == .Debug;
const BuddyAllocator = @import("buddy.zig").BuddyAllocator;
const Son = @This();
const Slices = BuddyAllocator(Id, Id.sentinel, 16, 1);
const IdRepr = u32;
const Slot = union { next: u32, elem: Node };

const sentinel = std.math.maxInt(u32);
const min_cap = 8;

pub const Fn = struct {
    entry: Id,
    exit: Id,
};

pub const Id = packed struct(IdRepr) {
    index: Index = 0,
    flag: std.meta.Tag(Kind) = 0,

    const Index = std.meta.Int(.unsigned, @bitSizeOf(IdRepr) - @bitSizeOf(Kind));

    const sentinel: Id = @bitCast(@as(IdRepr, std.math.maxInt(IdRepr)));
    const uninit: Id = @bitCast(@as(IdRepr, std.math.minInt(IdRepr)));

    pub fn lessThen(self: Id, other: Id) bool {
        return self.repr() < other.repr();
    }

    pub fn init(knd: Kind, index: usize) Id {
        return .{ .index = @intCast(index), .flag = @intFromEnum(knd) };
    }

    pub fn invalid(knd: Kind) Id {
        return init(knd, std.math.maxInt(Index));
    }

    pub fn isInvalid(self: Id) bool {
        return self.index == std.math.maxInt(Index);
    }

    pub fn kind(self: Id) Kind {
        return @enumFromInt(self.flag);
    }

    pub fn repr(self: Id) IdRepr {
        return @bitCast(self);
    }

    pub fn eql(self: Id, other: Id) bool {
        return self.repr() == other.repr();
    }

    pub fn format(
        self: Id,
        comptime ft: []const u8,
        options: std.fmt.FormatOptions,
        writer: anytype,
    ) !void {
        _ = options;
        _ = ft;
        try writer.print("{s}{d}", .{ @tagName(self.kind()), self.index });
    }
};

pub const Slice = struct {
    data: union {
        base: Slices.Index,
        item: Id,
    } = .{ .item = undefined },
    meta: packed struct(u32) {
        len: Slices.Size = 0,
        cap_pow2: Slices.SClass = undefined,
        depth: u12 = 0,
    } = .{},

    pub fn view(self: *Slice, slices: Slices) []Id {
        return switch (self.meta.len) {
            0 => &.{},
            1 => @as([*]Id, @ptrCast(&self.data.item))[0..1],
            else => slices.mem[self.data.base..][0..self.meta.len],
        };
    }

    pub fn append(self: *Slice, gpa: std.mem.Allocator, slices: *Slices, value: Id) !void {
        switch (self.meta.len) {
            0 => self.data = .{ .item = value },
            1 => {
                const idx = try slices.alloc(gpa, 2);
                slices.mem[idx..][0..2].* = .{ self.data.item, value };
                self.data = .{ .base = idx };
                self.meta.cap_pow2 = Slices.sclassOf(2);
            },
            else => {
                const cap = Slices.sizeOf(self.meta.cap_pow2);
                if (self.meta.len == cap) {
                    if (!slices.grow(self.data.base, self.meta.len)) {
                        const idx = try slices.alloc(gpa, self.meta.len * 2);
                        @memcpy(slices.mem.ptr[idx..], slices.mem[self.data.base..][0..self.meta.len]);
                        slices.free(self.data.base, self.meta.len);
                        self.data.base = idx;
                    }
                    self.meta.cap_pow2 += 1;
                }
                slices.mem[self.data.base + self.meta.len] = value;
            },
        }
        self.meta.len += 1;
        if (debug) for (self.view(slices.*)) |el| std.debug.assert(!el.eql(Id.sentinel));
    }

    pub fn pop(self: *Slice, slices: *Slices) Id {
        const vew = self.view(slices.*);
        const value = vew[vew.len - 1];
        const cap = Slices.sizeOf(self.meta.cap_pow2);
        switch (self.meta.len) {
            0 => unreachable,
            1 => {},
            2 => {
                const last_value = vew[0];
                slices.free(self.data.base, 2);
                self.data = .{ .item = last_value };
            },
            else => if (self.meta.len - 1 == cap / 2) {
                slices.shrink(self.data.base, @intCast(cap));
                self.meta.cap_pow2 -= 1;
            },
        }
        self.meta.len -= 1;
        return value;
    }

    pub fn remove(self: *Slice, slices: *Slices, value: Id) void {
        const vew = self.view(slices.*);
        const index = std.mem.indexOfScalar(IdRepr, @ptrCast(vew), value.repr()).?;
        std.mem.swap(Id, &vew[index], &vew[vew.len - 1]);
        _ = self.pop(slices);
    }

    pub inline fn len(self: Slice) usize {
        return self.meta.len;
    }
};

pub const Kind = enum {
    // NOTE: ordering is deliberate (see Parser.peepholeBinOp)
    const_int,
    cfg_start,
    cfg_tuple,
    cfg_if,
    @"cfg_if:true",
    @"cfg_if:false",
    cfg_region,
    cfg_return,
    cfg_end,
    phi,
    @"uo-",
    @"bo+",
    @"bo-",
    @"bo*",
    @"bo/",

    pub fn isCommutative(self: Kind) bool {
        return switch (self) {
            .@"bo+", .@"bo*" => true,
            else => false,
        };
    }

    pub fn isConst(self: Kind) bool {
        return switch (self) {
            inline else => |s| return comptime std.mem.startsWith(u8, @tagName(s), "const_"),
        };
    }

    pub fn isCfg(self: Kind) bool {
        return switch (self) {
            inline else => |s| return comptime std.mem.startsWith(u8, @tagName(s), "cfg_"),
        };
    }

    pub fn isBinOp(self: Kind) bool {
        return switch (self) {
            inline else => |s| return comptime std.mem.eql(u8, s.inputPayloadName(), "bo"),
        };
    }

    pub fn applyUnOp(comptime self: Kind, oper: anytype) @TypeOf(oper) {
        return switch (self) {
            .@"uo-" => -%oper,
            else => @compileError("wat"),
        };
    }

    pub fn applyBinOp(comptime self: Kind, lhs: anytype, rhs: @TypeOf(lhs)) @TypeOf(lhs) {
        return switch (self) {
            .@"bo+" => lhs +% rhs,
            .@"bo-" => lhs -% rhs,
            .@"bo*" => lhs *% rhs,
            .@"bo/" => @divFloor(lhs, rhs),
            else => @compileError("wat"),
        };
    }

    pub fn inputPayloadName(self: Kind) []const u8 {
        return for (@tagName(self), 0..) |c, i| switch (c) {
            'a'...'z', '_' => {},
            else => break @tagName(self)[0..i],
        } else @tagName(self);
    }

    pub fn InputPayload(comptime self: Kind) type {
        const name = self.inputPayloadName();
        return @TypeOf(@field(@as(Inputs, undefined), name));
    }
};

pub const Return = extern struct { cfg: Id, end: Id, value: Id };
pub const BinOp = extern struct { lhs: Id, rhs: Id };
pub const UnOp = extern struct { oper: Id };
pub const Int = extern struct { value: i64 };
pub const Tuple = extern struct { cfg: Id, index: u32 };
pub const If = extern struct { cfg: Id, cond: Id };
pub const Phi = extern struct { region: Id, left: Id, right: Id };
pub const Region = extern struct { lcfg: Id, rcfg: Id };

pub const Inputs = union {
    cfg_start: extern struct {},
    cfg_tuple: Tuple,
    cfg_if: If,
    cfg_region: Region,
    cfg_return: Return,
    cfg_unreachable: extern struct {},
    cfg_end: extern struct {},
    phi: Phi,
    const_int: Int,
    uo: UnOp,
    bo: BinOp,

    fn init(payload: anytype) Inputs {
        return @unionInit(Inputs, nameForPayload(@TypeOf(payload)), payload);
    }

    pub fn nameForPayload(comptime P: type) []const u8 {
        inline for (@typeInfo(Inputs).Union.fields) |field| {
            if (field.type == P) return field.name;
        }
        @compileError("Inputs.init: unrecognized payload: " ++ @typeName(P));
    }

    pub fn idsOf(self: *const Inputs, comptime payload: []const u8) *const [countIds(payload)]Id {
        if (comptime countIds(payload) == 0) return undefined;
        return @ptrCast(&@field(self, payload));
    }

    pub fn idsOfPayload(payload: anytype) *const [countIds(nameForPayload(@TypeOf(payload.*)))]Id {
        return @ptrCast(payload);
    }

    fn countIds(comptime payload: []const u8) usize {
        const P = @TypeOf(@field(@as(Inputs, undefined), payload));
        var count: usize = 0;
        std.debug.assert(@typeInfo(P).Struct.layout == .@"extern");
        for (@typeInfo(P).Struct.fields) |field| {
            if (field.type == Id) count += 1 else break;
        }
        for (@typeInfo(P).Struct.fields[count..]) |field| std.debug.assert(field.type != Id);
        return count;
    }
};

pub const Node = struct {
    kind: if (debug) Kind else void = undefined,
    refs: Slice = .{},
    inputs: Inputs,
};

fn dbg(any: anytype) @TypeOf(any) {
    std.debug.print("{any}\n", .{any});
    return any;
}

pub const Fmt = struct {
    son: *Son,
    out: *std.ArrayList(u8),
    visited: std.ArrayListUnmanaged(Id) = .{},

    pub fn deinit(self: *Fmt) void {
        self.visited.deinit(self.out.allocator);
    }

    pub fn fmt(self: *Fmt, id: Id) !void {
        if (std.mem.indexOfScalar(u32, @ptrCast(self.visited.items), @bitCast(id)) != null)
            return;
        try self.visited.append(self.out.allocator, id);
        const nd = self.son.getPtr(id);

        switch (id.kind()) {
            inline else => |t| {
                const payload = @field(nd.inputs, t.inputPayloadName());
                if (@TypeOf(payload) != void) {
                    inline for (@typeInfo(@TypeOf(payload)).Struct.fields) |field| {
                        if (field.type != Id or comptime std.mem.indexOf(u8, field.name, "cfg") != null) continue;
                        try self.fmt(@field(payload, field.name));
                    }

                    try self.out.writer().print("{any}:", .{id});
                    inline for (@typeInfo(@TypeOf(payload)).Struct.fields) |field| {
                        try self.out.writer().print(" {any}", .{@field(payload, field.name)});
                    }
                } else {
                    try self.out.writer().print("{any}: void", .{id});
                }

                try self.out.append('\n');

                if (comptime t.isCfg()) {
                    const outs = nd.refs.view(self.son.slices);
                    for (outs) |out| if (out.kind().isCfg()) try self.fmt(out);
                }
            },
        }
    }
};

pub fn deinit(self: *Son, gpa: std.mem.Allocator) void {
    self.slices.deinit(gpa);
    gpa.free(self.nodes);
    self.* = undefined;
}

pub fn get(self: Son, id: Id) Node {
    return self.nodes[id.index].elem;
}

pub fn getPtr(self: *Son, id: Id) *Node {
    return &self.nodes[id.index].elem;
}

pub fn add(self: *Son, gpa: std.mem.Allocator, comptime kind: Kind, inputs: kind.InputPayload()) !Id {
    if (self.free == sentinel) {
        const new_len = @max(self.nodes.len * 2, min_cap);
        const old_len = self.nodes.len;
        self.nodes = try gpa.realloc(self.nodes, new_len);
        var i = self.nodes.len;
        while (old_len < i) {
            i -= 1;
            self.nodes[i] = .{ .next = self.free };
            self.free = @intCast(i);
        }
    }

    const index = self.free;
    self.free = self.nodes[index].next;
    self.nodes[index] = .{ .elem = .{ .inputs = Inputs.init(inputs) } };
    if (debug) self.nodes[index].elem.kind = kind;
    return Id.init(kind, index);
}

pub fn rmeove(self: *Son, id: Id) void {
    std.debug.assert(self.nodes[id.index].elem.refs.len() == 0);
    if (debug) std.debug.assert(self.nodes[id.index].elem.kind == id.kind());
    self.nodes[id.index] = .{ .next = self.free };
    self.free = @intCast(id.index);
}

pub fn fmt(self: *Son, root: Id, out: *std.ArrayList(u8)) !void {
    var fm = Fmt{ .son = self, .out = out };
    defer fm.deinit();
    try fm.fmt(root);
}

pub fn collectLeakedIds(self: *const Son, root: Id, into: *std.ArrayList(Id)) !void {
    try into.resize(self.nodes.len);
    for (into.items, 0..) |*elem, i| elem.* = Id.init(.cfg_start, i);

    var cursor = self.free;
    while (cursor != sentinel) : (cursor = self.nodes[cursor].next) into.items[cursor] = Id.invalid(.cfg_start);

    self.eraseReachable(root, into.items);

    var write: usize = 0;
    for (into.items) |it| {
        if (it.isInvalid()) continue;
        into.items[write] = Id.init(self.nodes[it.index].elem.kind, it.index);
        write += 1;
    }
    into.items.len = write;
}

fn eraseReachable(self: *const Son, id: Id, visited: []Id) void {
    if (visited[id.index].isInvalid()) return;
    visited[id.index] = Id.invalid(.cfg_start);
    var nd = self.get(id);
    const outs = nd.refs.view(self.slices);
    for (outs) |out| self.eraseReachable(out, visited);
    switch (id.kind()) {
        inline else => |t| {
            for (Inputs.idsOf(&nd.inputs, t.inputPayloadName())) |it|
                self.eraseReachable(it, visited);
        },
    }
}
