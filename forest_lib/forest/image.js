"use strict";
// bokeh.models.CustomJS callback code
//
// args:
//      span - bokeh.models.Span
//      left_images - ColumnDataSource
//      left_alpha - Copy of left_images alpha values
//      right_images - ColumnDataSource
//      right_alpha - Copy of right_images alpha values
//      shared - ColumnDataSource
// cb_data:
//      geometry.x - mouse x position relative to figure
//
let reveal_image = function(side,
                            source,
                            mouse_x,
                            previous_mouse_x,
                            use_previous_mouse_x) {
    // RGBA image extents in mouse position space
    let x = source.data["x"][0];
    let y = source.data["y"][0];
    let dw = source.data["dw"][0];
    let dh = source.data["dh"][0];
    let original_alpha = source.data["_alpha"][0];
    let ni = source.data["_shape"][0][0];
    let nj = source.data["_shape"][0][1];

    // Useful debug information
    let mode = "silent";
    if (mode === "debug") {
        console.log(side, "x", x);
        console.log(side, "y", y);
        console.log(side, "dw", dw);
        console.log(side, "dh", dh);
        console.log(side, "alpha", original_alpha[0]);
        console.log(side, "ni", ni);
        console.log(side, "nj", nj);
    }

    // Mouse position(s)
    let left_x;
    let right_x;
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
    let original_alpha_value;
    let dy = dw / nj;
    let skip = 0;

    for (let j=0; j<nj; j++) {
        pixel_x = x + (j * dy);

        // Optimised selection of columns between mouse events
        // note: feature turned off during first paint
        if (use_previous_mouse_x) {
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
            if (visible_pixel(pixel_x, mouse_x, side)) {
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
};

let visible_pixel = function(pixel_x, mouse_x, show_side) {
    if (show_side === "left") {
        return pixel_x < mouse_x;
    } else {
        return pixel_x > mouse_x;
    }
};

// CustomJS callback main program
let main = function(cb_data,
                    left_images,
                    right_images,
                    shared,
                    span) {
    // Gather data from cb_data and args
    let mouse_x = cb_data.geometry.x;
    let previous_mouse_x = shared.data.previous_mouse_x[0];
    let use_previous_mouse_x = shared.data.use_previous_mouse_x[0];

    // Move vertical line to mouse position
    span.location = mouse_x;

    // Update image alpha values
    reveal_image("left",
                 left_images,
                 mouse_x,
                 previous_mouse_x,
                 use_previous_mouse_x);
    reveal_image("right",
                 right_images,
                 mouse_x,
                 previous_mouse_x,
                 use_previous_mouse_x);

    // Update shared data
    shared.data.previous_mouse_x[0] = mouse_x;
    shared.data.use_previous_mouse_x[0] = true;
};

if (typeof module === 'undefined') {
    // Bokeh call back usage
    main(cb_data,
         left_images,
         right_images,
         shared,
         span);
} else {
    // NPM test usage
    module.exports = {
        main: main,
        reveal_image: reveal_image,
        visible_pixel: visible_pixel
    };
}
