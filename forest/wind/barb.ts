import {RenderOne} from "models/markers/defs"
import {Marker, MarkerView} from "models/markers/marker"
import {Class} from "core/class"
import {Line, Fill} from "core/visuals"
import {Context2d} from "core/util/canvas"
// import * as p from "core/properties"
//declare var barbs: any;
import * as barbs from './lib/barbs'


function _one_barb(
        ctx: Context2d,
        i: number,
        r: number,
        line: Line,
        fill: Fill): void {
      console.log(i, r)
      barbs.draw(
          ctx,
          10,
          10,
          r
      )
//        barbs.draw(
//            ctx,
//            this._u[i],
//            this._v[i],
//            r
//        )
    if (fill.doit) {
        fill.set_vectorize(ctx, i)
        ctx.fill()
    }
    if (line.doit) {
        line.set_vectorize(ctx, i)
        ctx.stroke()
    }
}


function _mk_model(f: RenderOne): Class<Marker> {
    // Replicate approach in markers/defs.ts

    const view = class extends MarkerView {
        static initClass(): void {
            this.prototype._render_one = f
        }
    }
    view.initClass()

    const model = class extends Marker {
        static __name__ = "Barb"
        static initClass(): void {
            this.prototype.default_view = view
        }
    }
    model.initClass()
    return model
}

export const Barb = _mk_model(_one_barb)

// Barb.define({
//     "u": [p.DistanceSpec,
//         {units: "screen", value: 0}],
//     "v": [p.DistanceSpec,
//         {units: "screen", value: 0}],
// })
