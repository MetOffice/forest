import {Marker, MarkerView} from "models/markers/marker"
import {Line, Fill} from "core/visuals"
import {Context2d} from "core/util/canvas"
import * as p from "core/properties"
import * as barbs from './dist/barbs'

class BarbView extends MarkerView {
    _render_one(
            ctx: Context2d,
            i: number,
            r: number,
            line: Line,
            fill: Fill): void {
        barbs.draw(
            ctx,
            this._u[i],
            this._v[i],
            r
        )
        if (fill.doit) {
            fill.set_vectorize(ctx, i)
            ctx.fill()
        }
        if (line.doit) {
            line.set_vectorize(ctx, i)
            ctx.stroke()
        }
    }
}

export class Barb extends Marker {
    default_view = BarbView
    type = "Barb"
}
Barb.define({
    "u": [p.DistanceSpec,
        {units: "screen", value: 0}],
    "v": [p.DistanceSpec,
        {units: "screen", value: 0}],
})
