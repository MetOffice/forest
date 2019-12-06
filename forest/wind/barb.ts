import {XYGlyph, XYGlyphView, XYGlyphData} from "models/glyphs/xy_glyph"
//import {PointGeometry} from "core/geometry"
import {PointGeometry, SpanGeometry, RectGeometry, PolyGeometry} from "core/geometry"
import {LineVector, FillVector} from "core/property_mixins"
import {Line, Fill} from "core/visuals"
import {Arrayable, Rect} from "core/types"
import * as hittest from "core/hittest"
import * as p from "core/properties"
import {range} from "core/util/array"
import {map} from "core/util/arrayable"
import {Context2d} from "core/util/canvas"
import {Selection} from "models/selections/selection"

//declare var barbs: any;
import * as barbs from './lib/barbs'

// This is silly, but I am going to use the RadiusDimension Type for our
// dimension, because it already exists in core/properties.
import {RadiusDimension} from "core/enums"
//export type BarbDimension = "x" | "y" | "max" | "min"
//export const BarbDimension: BarbDimension[] = ["x", "y", "max", "min"]

export interface BarbData extends XYGlyphData {
  _u: Arrayable<number>
  _v: Arrayable<number>
  su: Arrayable<number>
  sv: Arrayable<number>
}

export interface BarbView extends BarbData {}

export class BarbView extends XYGlyphView {
  model: Barb
  visuals: Barb.Visuals

  protected _map_data(): void {
    console.log("_map_data")
    // XXX: Order is important here: size is always present (at least
    // a default), but radius is only present if a user specifies it.
    const bd = this.model.properties.barb_dimension.spec.value
    switch (bd) {
      case "x": {
        this.su = this.sdist(this.renderer.xscale, this._x, this._u)
        this.sv = this.sdist(this.renderer.yscale, this._x, this._v)
        break
      }
      case "y": {
        this.su = this.sdist(this.renderer.xscale, this._y, this._u)
        this.sv = this.sdist(this.renderer.yscale, this._y, this._v)
        break
      }
      case "max": {
        const u = this.sdist(this.renderer.xscale, this._x, this._u)
        const v = this.sdist(this.renderer.yscale, this._y, this._v)
        this.su = map(u, (s, i) => Math.max(s, u[i]))
        this.sv = map(v, (s, i) => Math.max(s, v[i]))
        break
      }
      case "min": {
        const u = this.sdist(this.renderer.xscale, this._x, this._u)
        const v = this.sdist(this.renderer.yscale, this._y, this._v)
        this.su = map(u, (s, i) => Math.min(s, u[i]))
        this.sv = map(v, (s, i) => Math.min(s, v[i]))
        break
      }
    }
    console.log("leave _map_data")
  }

  protected _mask_data(): number[] {
    console.log("_mask_data")
    const [hr, vr] = this.renderer.plot_view.frame.bbox.ranges

    let x0: number, y0: number
    let x1: number, y1: number
    const sx0 = hr.start
    const sx1 = hr.end
    ;[x0, x1] = this.renderer.xscale.r_invert(sx0, sx1)
    x0 -= 1
    x1 += 1

    const sy0 = vr.start
    const sy1 = vr.end
    ;[y0, y1] = this.renderer.yscale.r_invert(sy0, sy1)
    y0 -= 1
    y1 += 1

    console.log("leave _mask_data")
    return this.index.indices({x0, x1, y0, y1})
  }

  protected _render(ctx: Context2d, indices: number[], {sx, sy, su, sv}: BarbData): void {
    for (const i of indices) {
      if (isNaN(su[i] + sv[i]))
        continue

      ctx.translate(sx[i], sy[i])
      barbs.draw(ctx, su[i], sv[i], 1)
      ctx.translate(-sx[i], -sy[i])
      console.log("su, sv values: ", su[i], sv[i])

      if (this.visuals.fill.doit) {
        this.visuals.fill.set_vectorize(ctx, i)
        ctx.fill()
      }

      if (this.visuals.line.doit) {
        this.visuals.line.set_vectorize(ctx, i)
        ctx.stroke()
      }
    }
    console.log("we left _render, still nothing")
  }

  protected _hit_point(geometry: PointGeometry): Selection {
    console.log("_hit_point")
    let dist, r2, sx0, sx1, sy0, sy1, x0, x1, y0, y1
    const {sx, sy} = geometry
    const x = this.renderer.xscale.invert(sx)
    const y = this.renderer.yscale.invert(sy)

    x0 = x - 1000
    x1 = x + 1000

    y0 = y - 1000
    y1 = y + 1000


    const candidates = this.index.indices({x0, x1, y0, y1})

    const hits: [number, number][] = []
    for (const i of candidates) {
      r2 = Math.pow(this.su[i], 2) + Math.pow(this.sv[i], 2)
      ;[sx0, sx1] = this.renderer.xscale.r_compute(x, this._x[i])
      ;[sy0, sy1] = this.renderer.yscale.r_compute(y, this._y[i])
      dist = Math.pow(sx0-sx1, 2) + Math.pow(sy0-sy1, 2)
      if (dist <= r2) {
        hits.push([i, dist])
      }
    }

    console.log("leave _hit_point")
    return hittest.create_hit_test_result_from_hits(hits)
  }

  protected _hit_span(geometry: SpanGeometry): Selection {
    console.log("_hit_span")
    const {sx, sy} = geometry
    const bounds = this.bounds()
    const result = hittest.create_empty_hit_test_result()

    let x0, x1, y0, y1
    if (geometry.direction == 'h') {
      // use circle bounds instead of current pointer y coordinates
      let sx0, sx1
      y0 = bounds.y0
      y1 = bounds.y1
      sx0 = sx - 1000
      sx1 = sx + 1000
      ;[x0, x1] = this.renderer.xscale.r_invert(sx0, sx1)
    } else {
      // use circle bounds instead of current pointer x coordinates
      let sy0, sy1
      x0 = bounds.x0
      x1 = bounds.x1
      sy0 = sy - 1000
      sy1 = sy + 1000
      ;[y0, y1] = this.renderer.yscale.r_invert(sy0, sy1)
    }

    const hits = this.index.indices({x0, x1, y0, y1})

    result.indices = hits
    console.log("leave _hit_span")
    return result
  }

  protected _hit_rect(geometry: RectGeometry): Selection {
    console.log("_hit_rect")
    const {sx0, sx1, sy0, sy1} = geometry
    const [x0, x1] = this.renderer.xscale.r_invert(sx0, sx1)
    const [y0, y1] = this.renderer.yscale.r_invert(sy0, sy1)
    const result = hittest.create_empty_hit_test_result()
    result.indices = this.index.indices({x0, x1, y0, y1})
    console.log("leave _hit_rect")
    return result
  }

  protected _hit_poly(geometry: PolyGeometry): Selection {
    console.log("_hit_poly")
    const {sx, sy} = geometry

    // TODO (bev) use spatial index to pare candidate list
    const candidates = range(0, this.sx.length)

    const hits = []
    for (let i = 0, end = candidates.length; i < end; i++) {
      const idx = candidates[i]
      if (hittest.point_in_poly(this.sx[i], this.sy[i], sx, sy)) {
        hits.push(idx)
      }
    }

    const result = hittest.create_empty_hit_test_result()
    result.indices = hits
    console.log("leave _hit_poly")
    return result
  }

  // barb does not inherit from marker (since it also accepts u and v) so we
  // must supply a draw_legend for it  here
  draw_legend_for_index(ctx: Context2d, {x0, y0, x1, y1}: Rect, index: number): void {
    // using objects like this seems a little wonky, since the keys are coerced to
    // stings, but it works
    console.log("_draw_legend_for_index")
    const len = index + 1

    const u: number[] = new Array(len)
    u[index] = (x0 + x1)/2
    const v: number[] = new Array(len)
    v[index] = (y0 + y1)/2

    this._render(ctx, [index], {u, v} as any) // XXX
    console.log("leave _draw_legend_for_index")
  }
}

export namespace Barb {
  export type Attrs = p.AttrsOf<Props>

  export type Props = XYGlyph.Props & LineVector & FillVector & {
    u: p.DistanceSpec
    v: p.DistanceSpec
    barb_dimension: p.Property<RadiusDimension>
  }

  export type Visuals = XYGlyph.Visuals & {line: Line, fill: Fill}
}

export interface Barb extends Barb.Attrs {}

export class Barb extends XYGlyph {
  properties: Barb.Props

  constructor(attrs?: Partial<Barb.Attrs>) {
    super(attrs)
  }

  static init_Barb(): void {
    this.prototype.default_view = BarbView

    this.mixins(['line', 'fill'])
    this.define<Barb.Props>({
      u: [ p.DistanceSpec,    { units: "screen", value: 0 } ],
      v: [ p.DistanceSpec,    { units: "screen", value: 0 } ],
      barb_dimension: [ p.RadiusDimension, 'x' ],
    })
  }

  initialize(): void {
    super.initialize()
  }
}
