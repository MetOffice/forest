#!/usr/bin/env python3
import numpy as np
import bokeh.io
import bokeh.plotting
import bokeh.models.callbacks
import imageio


def main(bokeh_id):
    """Main program"""
    figure = bokeh.plotting.figure(sizing_mode="stretch_both", match_aspect=True)

    # High-level user interface
    wind_rgba = imageio.imread("windmill.png")[::-1, :, :]
    forest_rgba = imageio.imread("forest.png")[::-1, :, :]
    tools = slider_tool(figure, wind_rgba, forest_rgba)
    figure.add_tools(*tools)

    if bokeh_id == '__main__':
        bokeh.plotting.show(figure)
    else:
        bokeh.io.curdoc().add_root(figure)

def slider_tool(figure, left_rgba, right_rgba):
    """SliderTool prototype"""
    # Left image
    source = bokeh.models.ColumnDataSource(dict(image=[left_rgba]))
    left = figure.image_rgba(image="image",
                             x=0,
                             y=0,
                             dw=10,
                             dh=10,
                             source=source)
    left_image_tool = hover_image_tool(source, mode="show_left")

    # Right image
    source = bokeh.models.ColumnDataSource(dict(image=[right_rgba]))
    right = figure.image_rgba(image="image",
                              x=0,
                              y=0,
                              dw=10,
                              dh=10,
                              source=source)
    right_image_tool = hover_image_tool(source, mode="show_right")

    # Hide right image initially
    source.data["image"][0][:, :, -1] = 0.

    # VLine
    vertical_line_tool = vertical_line(figure, location=10)
    return left_image_tool, right_image_tool, vertical_line_tool

def hover_image_tool(source, mode):
    """Hide anything to the left/right of pointer

    At the moment this is achieved through the use of CustomJS and
    alpha values. A more complete solution would work on the canvas
    itself
    """
    code_template = """
        // Hard-coded values for now
        let x = 0;
        let dw = 10;

        // Shared data
        let original_alpha = shared.data.original_alpha[0];
        let previous_mouse_x = shared.data.mouse_x[0];
        let first_time = shared.data.first_time[0]
        let shape = shared.data.shape[0];
        let ni = shape[0];
        let nj = shape[1];

        // Mouse position(s)
        let left_x;
        let right_x;
        let mouse_x = cb_data.geometry.x;
        if (!isFinite(mouse_x)) {
            return;
        }
        if (mouse_x > previous_mouse_x) {
            left_x = previous_mouse_x;
            right_x = mouse_x;
        } else {
            left_x = mouse_x;
            right_x = previous_mouse_x;
        }

        // Update alpha pseudo-2D and RGBA pseudo-3D arrays
        let pixel_x;
        let alpha;
        let alpha_index;
        let image_alpha_index;
        let dy = dw / nj;
        let skip = 0;
        for (let j=0; j<nj; j++) {
            pixel_x = x + (j * dy);

            // Optimised selection of columns between mouse events
            // note: feature turned off during first paint
            if (!first_time) {
                if ((pixel_x > right_x) || (pixel_x < left_x)) {
                    // pixel outside current and previous mouse positions
                    skip += 1;
                    continue;
                }
            }

            // Ordinary loop logic
            for (let i=0; i<ni; i++) {
                alpha_index = (nj * i) + j;
                original_alpha_value = original_alpha[alpha_index];
                if (original_alpha_value == 0) {
                    continue;
                }
                image_alpha_index = (4 * alpha_index) + 3;
                if (%s) {
                   alpha = original_alpha_value;
                } else {
                   alpha = 0;
                }
               source.data["image"][0][image_alpha_index] = alpha;
            }
        }
        if (skip !== nj) {
            // Some columns need to be painted
            source.change.emit();
        }

        // Update shared data
        shared.data.mouse_x[0] = mouse_x;
        shared.data.first_time[0] = false;
    """
    if mode == "show_left":
        show_logic = "pixel_x < mouse_x"
    else:
        show_logic = "pixel_x > mouse_x"
    code = code_template % show_logic

    # Shared data needed to implement slider
    rgba = source.data["image"][0]
    original_alpha = np.copy(rgba[:, :, -1])
    shared = bokeh.models.ColumnDataSource(dict(mouse_x=[0],
                                                first_time=[True],
                                                shape=[rgba.shape],
                                                original_alpha=[original_alpha]))

    callback = bokeh.models.callbacks.CustomJS(args=dict(source=source,
                                                         shared=shared),
                                               code=code)
    return bokeh.models.HoverTool(callback=callback)

def windy_forest(figure):
    left_source = bokeh.models.ColumnDataSource({
        "x": [1, 2, 3],
        "y": [1, 2, 3],
        "width": [1, 1, 1],
        "alpha": [1, 1, 1]
    })
    left_renderer = figure.rect(x="x",
                                y="y",
                                width="width",
                                height=1,
                                color="blue",
                                alpha="alpha",
                                source=left_source)
    hover_tool = hover_tool_hide(left_source, side="right")
    figure.add_tools(hover_tool)
    right_source = bokeh.models.ColumnDataSource({
        "x": [1, 2, 3],
        "y": [3, 2, 1],
        "width": [0.5, 0.5, 0.5],
        "alpha": [1, 1, 1]
    })
    right_renderer = figure.rect(x="x",
                                 y="y",
                                 width="width",
                                 height=0.5,
                                 color="red",
                                 alpha="alpha",
                                 source=right_source)
    hover_tool = hover_tool_hide(right_source, side="left")
    figure.add_tools(hover_tool)
    vline_tool = vertical_line(figure)
    figure.add_tools(vline_tool)

def hover_tool_hide(source, side="left"):
    """Hide anything to the left/right of pointer

    At the moment this is achieved through the use of CustomJS and
    alpha values. A more complete solution would work on the canvas
    itself
    """
    code_template = """
        let x_left, x_right;
        let x = source.data["x"];
        let width = source.data["width"];
        let mouse_x = cb_data.geometry.x;
        let alpha = [];
        for (let i=0; i<x.length; i++) {
            x_left = x[i] - (width[i] / 2);
            x_right = x[i] + (width[i] / 2);
            if (%s) {
               alpha.push(0);
            } else {
               alpha.push(1);
            }
        }
        source.data.alpha = alpha;
        source.change.emit();
    """
    if side.lower() == "left":
        code = code_template % "mouse_x < x_right"
    else:
        code = code_template % "mouse_x > x_left"
    callback = bokeh.models.callbacks.CustomJS(args=dict(source=source),
                                               code=code)
    return bokeh.models.HoverTool(callback=callback)


def vertical_line(figure, location=0):
    """Add vertical Span that follows mouse pointer"""
    span = bokeh.models.Span(location=location, dimension='height', line_color='black', line_width=1)
    figure.renderers.append(span)
    callback = bokeh.models.callbacks.CustomJS(args=dict(span=span), code="""
        span.location = cb_data.geometry.x;
    """)
    return bokeh.models.HoverTool(callback=callback)


main(__name__)
