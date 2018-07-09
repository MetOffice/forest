"use strict";
var expect = require('chai').expect;
var forest = require('./forest');
describe("visible_pixel", function() {
    let check_visible_pixel;
    beforeEach(function() {
        check_visible_pixel = function(pixel_x,
                                       mouse_x,
                                       side,
                                       expected) {
            let actual = forest.visible_pixel(pixel_x, mouse_x, side);
            expect(actual).to.be.equal(expected);
        };
    });
    it("should mark left pixel true given left", function() {
        check_visible_pixel(0, 1, "left", true);
    });
    it("should mark right pixel false given left", function() {
        check_visible_pixel(1.1, 1, "left", false);
    });
    it("should mark left pixel false given right", function() {
        check_visible_pixel(0, 1, "right", false);
    });
    it("should mark right pixel true given right", function() {
        check_visible_pixel(1.1, 1, "right", true);
    });
});
