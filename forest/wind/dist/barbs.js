(function (factory) {
    if (typeof module === "object" && typeof module.exports === "object") {
        var v = factory(require, exports);
        if (v !== undefined) module.exports = v;
    }
    else if (typeof define === "function" && define.amd) {
        define(["require", "exports"], factory);
    }
})(function (require, exports) {
    "use strict";
    Object.defineProperty(exports, "__esModule", { value: true });
    exports.draw = function (ctx, u, v, scale) {
        if (scale === void 0) { scale = 1; }
        var length = 7;
        var radius = length * 0.15;
        var c = exports.speed(u, v);
        var angle = exports.direction(u, v);
        ctx.beginPath();
        if (c < 5) {
            ctx.arc(0, 0, scale * radius, 0, 2 * Math.PI);
        }
        else {
            ctx.rotate(-angle);
            var x = void 0, y = void 0;
            var pts = exports.vertices(exports.count_tails(c));
            for (var j = 0; j < pts.length; j++) {
                x = scale * pts[j][0];
                y = -scale * pts[j][1];
                if (j === 0) {
                    ctx.moveTo(x, y);
                }
                else {
                    ctx.lineTo(x, y);
                }
            }
            ctx.rotate(angle);
        }
        ctx.closePath();
    };
    exports.vertices = function (arrow, height, length, spacing) {
        if (height === void 0) { height = 2.8; }
        if (length === void 0) { length = 7; }
        if (spacing === void 0) { spacing = 0.875; }
        // Special case for lone half barb
        if ((arrow.flags === 0) &&
            (arrow.full_barbs === 0) &&
            (arrow.half_barbs === 1)) {
            var position_1 = -length + (1.5 * spacing);
            return [
                [0, 0],
                [-length, 0],
                [position_1, 0],
                [position_1 - (spacing / 2), height / 2],
                [position_1, 0],
                [0, 0]
            ];
        }
        var pts = [];
        var position = -length;
        pts.push([0, 0]);
        if (arrow.flags > 0) {
            for (var ib = 0; ib < arrow.flags; ib++) {
                pts.push([position, 0]);
                pts.push([position + spacing, height]);
                pts.push([position + (2 * spacing), 0]);
                position += 2 * spacing;
                if (ib === (arrow.flags - 1)) {
                    position += spacing;
                }
                else {
                    position += (spacing / 2);
                }
            }
        }
        if (arrow.full_barbs > 0) {
            for (var ib = 0; ib < arrow.full_barbs; ib++) {
                pts.push([position, 0]);
                pts.push([position - spacing, height]);
                pts.push([position, 0]);
                position += spacing;
            }
        }
        if (arrow.half_barbs > 0) {
            pts.push([position, 0]);
            pts.push([position - (spacing / 2), height / 2]);
            pts.push([position, 0]);
            position += spacing;
        }
        pts.push([0, 0]);
        return pts;
    };
    exports.count_tails = function (speed) {
        var flags, full_barbs, half_barbs;
        flags = ~~(speed / 50);
        if (flags > 0) {
            speed = speed - (flags * 50);
        }
        full_barbs = ~~(speed / 10);
        if (full_barbs > 0) {
            speed = speed - (full_barbs * 10);
        }
        half_barbs = ~~(speed / 5);
        return {
            'flags': flags,
            'full_barbs': full_barbs,
            'half_barbs': half_barbs
        };
    };
    exports.speed = function (u, v) {
        return Math.sqrt(Math.pow(u, 2) + Math.pow(v, 2));
    };
    exports.direction = function (u, v) {
        return Math.atan2(v, u);
    };
});
