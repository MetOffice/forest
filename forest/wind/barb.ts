import {XYGlyph, XYGlyphView, XYGlyphData} from "models/glyphs/xy_glyph"
//import {PointGeometry} from "core/geometry"
//import {PointGeometry, SpanGeometry, RectGeometry, PolyGeometry} from "core/geometry"
import {LineVector, FillVector} from "core/property_mixins"
import {Line, Fill} from "core/visuals"
import {Arrayable, Rect} from "core/types"
//import * as hittest from "core/hittest"
import * as p from "core/properties"
//import {range} from "core/util/array"
//import {map} from "core/util/arrayable"
import {Context2d} from "core/util/canvas"
//import {Selection} from "models/selections/selection"

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
}

export interface BarbView extends BarbData {}

export class BarbView extends XYGlyphView {
  model: Barb
  visuals: Barb.Visuals

  protected _render(ctx: Context2d, indices: number[], {sx, sy}: BarbData): void {
    for (const i of indices) {
      if (isNaN(this._u[i] + this._v[i]))
        continue

      ctx.translate(sx[i], sy[i])
      barbs.draw(ctx, this._u[i], this._v[i], 5)
      ctx.translate(-sx[i], -sy[i])

      if (this.visuals.fill.doit) {
        this.visuals.fill.set_vectorize(ctx, i)
        ctx.fill()
      }

      if (this.visuals.line.doit) {
        this.visuals.line.set_vectorize(ctx, i)
        ctx.stroke()
      }
    }
  }

  // barb does not inherit from marker (since it also accepts u and v) so we
  // must supply a draw_legend for it  here
  // This code is copied from circle.ts, and so far it doesn't seem to be used.
  // Definitely u and v not what you expect.
  draw_legend_for_index(ctx: Context2d, {x0, y0, x1, y1}: Rect, index: number): void {
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
    u: p.NumberSpec
    v: p.NumberSpec
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
      u: [ p.NumberSpec,    { value: 0 } ],
      v: [ p.NumberSpec,    { value: 0 } ],
      barb_dimension: [ p.RadiusDimension, 'x' ],
    })
  }

  initialize(): void {
    super.initialize()
  }
}
