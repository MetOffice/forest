"use strict";
// JavaScript callback functions
let forest = {};
// boolean check of pixel position relative to mouse
forest.visible_pixel = function(pixel_x, mouse_x, show_side) {
    if (show_side === "left") {
        return pixel_x < mouse_x;
    } else {
        return pixel_x > mouse_x;
    }
};
module.exports = forest;
