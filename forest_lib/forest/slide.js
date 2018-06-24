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
let slide_image = function(source,
                           original_alpha,
                           mouse_x,
                           previous_mouse_x,
                           first_time,
                           shape,
                           side) {
    // Hard-coded values for now
    let x = 0;
    let dw = 10;

    // Shared data
    let ni = shape[0];
    let nj = shape[1];

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
let main = function() {
    // Gather data from cb_data and args
    let first_time = shared.data.first_time[0];
    let previous_mouse_x = shared.data.previous_mouse_x[0];
    let mouse_x = cb_data.geometry.x;

    // Move vertical line to mouse position
    span.location = mouse_x;

    // Left image extra data
    let left_alpha = left_extra.data["alpha"][0];
    let left_shape = left_extra.data["shape"][0];

    // Update image alpha values
    slide_image(left_images,
                left_alpha,
                mouse_x,
                previous_mouse_x,
                first_time,
                left_shape,
                "left");

    // Right image extra data
    let right_alpha = right_extra.data["alpha"][0];
    let right_shape = right_extra.data["shape"][0];

    // Update image alpha values
    slide_image(right_images,
                right_alpha,
                mouse_x,
                previous_mouse_x,
                first_time,
                right_shape,
                "right");

    // Update shared data
    shared.data.previous_mouse_x[0] = mouse_x;
    shared.data.first_time[0] = false;
};

main();
