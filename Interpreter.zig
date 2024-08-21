son: *Son,
computed: []?i64,
fuel: usize = undefined,

const std = @import("std");
const main = @import("main.zig");
const Lexer = @import("Lexer.zig");
const Token = Lexer.Token;
const Son = @import("Son.zig");
const Kind = Son.Kind;
const Fn = Son.Fn;
const Node = Son.Node;
const Id = Son.Id;
const Interpreter = @This();

pub fn init(son: *Son, gpa: std.mem.Allocator) !Interpreter {
    const computed = try gpa.alloc(?i64, son.nodes.len);
    @memset(computed, null);
    return .{
        .son = son,
        .computed = computed,
    };
}

pub fn deinit(self: *Interpreter, gpa: std.mem.Allocator) void {
    gpa.free(self.computed);
    self.* = undefined;
}

pub fn run(self: *Interpreter, root: Id) !i64 {
    var prev_control: Id = undefined;
    var control = root;
    for (0..100) |_| {
        const node = self.son.getPtr(control);
        const refs = node.refs.view(self.son.slices);
        //self.son.logNode(control);
        const next_control = switch (control.kind()) {
            .cfg_tuple, .cfg_start => refs[0],
            .@"cfg_region:loop", .cfg_region => b: {
                const left_side = node.inputs.cfg_region.lcfg.eql(prev_control);
                std.debug.assert(left_side or node.inputs.cfg_region.rcfg.eql(prev_control));
                var next: Id = undefined;
                for (refs) |phi| {
                    if (phi.kind() != .phi) {
                        next = phi;
                        continue;
                    }
                    self.clearCompute(phi);
                    const phi_node = self.son.getPtr(phi);
                    const inputs = phi_node.inputs.phi;
                    self.computed[phi.index] = self.eval(if (left_side) inputs.left else inputs.right);
                }
                break :b next;
            },
            .cfg_if => b: {
                const cond = self.eval(node.inputs.cfg_if.cond);
                break :b refs[@intFromBool(cond == 0)];
            },
            .cfg_return => return self.eval(node.inputs.cfg_return.value),
            else => |e| std.debug.panic("unhandled control frlow: {s}", .{@tagName(e)}),
        };
        prev_control = control;
        control = next_control;
    }

    @memset(self.computed, null);
    return error.Timeout;
}

fn eval(self: *Interpreter, id: Id) i64 {
    if (self.computed[id.index]) |v| return v;

    const node = self.son.getPtr(id);
    self.computed[id.index] = switch (id.kind()) {
        .const_int => node.inputs.const_int.value,
        inline else => |t| if (comptime t.isBinOp())
            t.applyBinOp(self.eval(node.inputs.bo.lhs), self.eval(node.inputs.bo.rhs))
        else
            std.debug.panic("undandled operator: {s}", .{@tagName(t)}),
    };
    return self.computed[id.index].?;
}

fn clearCompute(self: *Interpreter, id: Id) void {
    for (self.son.getPtr(id).refs.view(self.son.slices)) |out| {
        if (out.kind() == .phi) continue;
        if (self.computed[out.index] == null) continue;
        self.computed[out.index] = null;
        self.clearCompute(out);
    }
}
