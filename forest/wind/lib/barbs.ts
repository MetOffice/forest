"use strict";

export interface Arrow {
    flags: number;
    full_barbs: number;
    half_barbs: number;
}

export const draw = function(ctx, u, v, scale=1) {
    let length = 7
    let radius = length * 0.15
    let c = speed(u, v)
    let angle = direction(u, v)
    ctx.beginPath()
    if (c < 5) {
        ctx.arc(0, 0, scale * radius, 0, 2 * Math.PI)
    } else {
        ctx.rotate(-angle)
        let x, y
        let pts = vertices(count_tails(c))
        for (let j=0; j<pts.length; j++) {
            x = scale * pts[j][0]
            y = -scale * pts[j][1]
            if (j === 0) {
                ctx.moveTo(x, y)
            } else {
                ctx.lineTo(x, y)
            }
        }
        ctx.rotate(angle)
    }
    ctx.closePath()
}


export const vertices = function(
    arrow: Arrow,
    height=2.8,
    length=7,
    spacing=0.875
) {

    // Special case for lone half barb
    if ((arrow.flags === 0) &&
        (arrow.full_barbs === 0) &&
        (arrow.half_barbs === 1)) {
        let position = -length + (1.5 *spacing)
        return [
            [0, 0],
            [-length, 0],
            [position, 0],
            [position - (spacing / 2), height / 2],
            [position, 0],
            [0, 0]
        ]
    }

    let pts = []
    let position = -length
    pts.push([0, 0])
    if (arrow.flags > 0) {
        for (let ib=0; ib<arrow.flags; ib++) {
            pts.push([position, 0])
            pts.push([position + spacing, height])
            pts.push([position + (2 * spacing), 0])
            position += 2 * spacing
            if (ib === (arrow.flags - 1)) {
                position += spacing
            } else {
                position += (spacing / 2)
            }
        }
    }
    if (arrow.full_barbs > 0) {
        for (let ib=0; ib<arrow.full_barbs; ib++) {
            pts.push([position, 0])
            pts.push([position - spacing, height])
            pts.push([position, 0])
            position += spacing
        }
    }
    if (arrow.half_barbs > 0) {
        pts.push([position, 0])
        pts.push([position - (spacing / 2), height/2])
        pts.push([position, 0])
        position += spacing
    }
    pts.push([0, 0])
    return pts
}

export const count_tails = function(speed: number) : Arrow {
    let flags, full_barbs, half_barbs
    flags = ~~(speed / 50)
    if (flags > 0) {
        speed = speed - (flags * 50)
    }
    full_barbs = ~~(speed / 10)
    if (full_barbs > 0) {
        speed = speed - (full_barbs * 10)
    }
    half_barbs = ~~(speed / 5)
    return {
        'flags': flags,
        'full_barbs': full_barbs,
        'half_barbs': half_barbs
    }
}
export const speed = function(u : number, v : number) : number {
    return Math.sqrt(u**2 + v**2)
}

export const direction = function(u : number, v : number) : number {
    return Math.atan2(v, u)
}
