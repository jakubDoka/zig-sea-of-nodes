source: []const u8,
index: usize = 0,

const Lexer = @This();
const std = @import("std");

pub const Lexeme = enum(u8) {
    Eof = 0,

    @"return",
    @"if",
    @"else",
    @"while",

    true,
    false,

    Int,
    Ident,

    @"=" = '=',
    @";" = ';',
    @"+" = '+',
    @"-" = '-',
    @"*" = '*',
    @"/" = '/',
    @"<" = '<',
    @"{" = '{',
    @"}" = '}',
    @"(" = '(',
    @")" = ')',

    @":=" = ':' + 128,
    @"==" = '=' + 128,
    @"!=" = '!' + 128,

    pub fn prec(self: Lexeme) u8 {
        return switch (self) {
            .@"=", .@":=" => 15,
            .@"==", .@"!=", .@"<" => 7,
            .@"+", .@"-" => 4,
            .@"*", .@"/" => 3,
            else => 254,
        };
    }

    pub inline fn isOp(self: Lexeme) bool {
        return self.prec() != 254;
    }
};

pub const Token = packed struct(u64) {
    lexeme: Lexeme,
    length: u24,
    offset: u32,

    pub fn view(self: Token, source: []const u8) []const u8 {
        return source[self.offset..][0..self.length];
    }
};

pub fn next(self: *Lexer) Token {
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

pub fn peekStr(source: []const u8, pos: usize) []const u8 {
    var self = Lexer{ .source = source, .index = pos };
    return self.next().view(source);
}
