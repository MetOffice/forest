import {XYGlyph, XYGlyphView, XYGlyphData} from "models/glyphs/xy_glyph"
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

export interface BarbData extends XYGlyphData {
  _u: Arrayable<number>
  _v: Arrayable<number>
}

export interface BarbView extends BarbData {}

export class BarbView extends XYGlyphView {
  model: Barb
  visuals: Barb.Visuals

  //protected _map_data(): void {
  //}

  //protected _mask_data(): number[] {
  //  return this.index.indices({x0, x1, y0, y1})
  //}

  protected _render(ctx: Context2d, indices: number[], {_u, _v}: BarbData): void {
    for (const i of indices) {
      if (isNaN(_u[i] + _v[i]))
        continue

      barbs.draw(ctx, _u[i], _v[i], 10)
      console.log("we tried to render but nothing shows up")

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

  //protected _hit_point(geometry: PointGeometry): Selection {
  //}

  //protected _hit_span(geometry: SpanGeometry): Selection {
  //}

  //protected _hit_rect(geometry: RectGeometry): Selection {
  //}

  //protected _hit_poly(geometry: PolyGeometry): Selection {
  //}

  // barb does not inherit from marker (since it also accepts radius) so we
  // must supply a draw_legend for it  here
  draw_legend_for_index(ctx: Context2d, {x0, y0, x1, y1}: Rect, index: number): void {
    // using objects like this seems a little wonky, since the keys are coerced to
    // stings, but it works
    const len = index + 1

    const u: number[] = new Array(len)
    u[index] = (x0 + x1)/2
    const v: number[] = new Array(len)
    v[index] = (y0 + y1)/2

    this._render(ctx, [index], {u, v} as any) // XXX
  }
}

export namespace Barb {
  export type Attrs = p.AttrsOf<Props>

  export type Props = XYGlyph.Props & LineVector & FillVector & {
    u: p.DistanceSpec
    v: p.DistanceSpec
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
    })
  }

  initialize(): void {
    super.initialize()
  }
}
