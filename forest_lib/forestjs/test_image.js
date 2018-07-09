"use strict";
var expect = require('chai').expect;
var image = require('../forest/image');
describe("image.js", function() {
    describe("main", function() {
        it("should work with empty images", function() {
            let shared = {
                data: {
                    first_time: [false],
                    previous_mouse_x: [0]
                }
            };
            let cb_data = {
                geometry: {
                    x: 0
                }
            };
            let span = {
                location: null
            };
            let empty_image = {
                data: {
                    x: [0],
                    y: [0],
                    dw: [1],
                    dh: [1],
                    _alpha: [[]],
                    _shape: [[0, 0]]
                }
            };
            let left_images = empty_image;
            let right_images = empty_image;
            image.main(cb_data,
                       left_images,
                       right_images,
                       shared,
                       span);
        });
        it("should work with 2x2 pixel images", function() {
            // Mouse-x should highlight left/right alpha values
            let mouse_x = 0;
            let shared = {
                data: {
                    first_time: [false],
                    previous_mouse_x: [0]
                }
            };
            let cb_data = {
                geometry: {
                    x: mouse_x
                }
            };
            let span = {
                location: null
            };
            let two_by_two = {
                data: {
                    x: [0],
                    y: [0],
                    dw: [1],
                    dh: [1],
                    image: [[1, 1, 1, 0,
                             1, 1, 1, 0,
                             1, 1, 1, 0,
                             1, 1, 1, 0]],
                    _alpha: [[0, 0, 0, 0]],
                    _shape: [[2, 2]]
                },
                change: {
                    emit: function() {}
                }
            };
            let left_images = two_by_two;
            let right_images = two_by_two;
            image.main(cb_data,
                       left_images,
                       right_images,
                       shared,
                       span);
            // Assertions missing
            expect(false).to.be.equal(true);
        });
    });
    describe("visible_pixel", function() {
        let check_visible_pixel;
        beforeEach(function() {
            check_visible_pixel = function(pixel_x,
                                           mouse_x,
                                           side,
                                           expected) {
                let actual = image.visible_pixel(pixel_x, mouse_x, side);
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
});
