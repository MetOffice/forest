"use strict";
var expect = require('chai').expect;
var image = require('../forest/image');
describe("image.js", function() {
    // Helper methods
    let two_by_two = null;
    let four_pixels = null;
    beforeEach(function() {
        // Creates 2x2 pixel image ColumnDataSource
        two_by_two = function() {
            return four_pixels([2, 2]);
        };

        // Create 4 pixels of customisable shape
        four_pixels = function(shape) {
            // function returns new Object
            return {
                data: {
                    x: [0],
                    y: [0],
                    dw: [1],
                    dh: [1],
                    image: [[1, 1, 1, 255,
                             1, 1, 1, 255,
                             1, 1, 1, 255,
                             1, 1, 1, 255]],
                    _alpha: [[255, 255, 255, 255]],
                    _shape: [shape]
                },
                change: {
                    emit: function() {}
                }
            };
        };
    });
    describe("main", function() {
        let shared = null;
        let span = null;
        let cb_data = null;
        beforeEach(function() {
            shared = {
                data: {
                    use_previous_mouse_x: [false],
                    previous_mouse_x: [0]
                }
            };
            cb_data = {
                geometry: {
                    x: null
                }
            };
            span = {
                location: null
            };
        });
        it("should work with empty images", function() {
            cb_data.geometry.x = 0;
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
            let left_images = two_by_two();
            let right_images = two_by_two();

            // System under test
            cb_data.geometry.x = 1;
            image.main(cb_data,
                       left_images,
                       right_images,
                       shared,
                       span);

            // Assertions
            let actual, expected;
            actual = left_images.data["image"][0];
            expected = [1, 1, 1, 255,
                        1, 1, 1, 255,
                        1, 1, 1, 255,
                        1, 1, 1, 255];
            expect(expected).to.be.deep.equal(actual);
            actual = right_images.data["image"][0];
            expected = [1, 1, 1, 0,
                        1, 1, 1, 0,
                        1, 1, 1, 0,
                        1, 1, 1, 0];
            expect(expected).to.be.deep.equal(actual);
        });
        it("should turn off right side of left_images", function() {
            let left_images = two_by_two();
            let right_images = two_by_two();

            // System under test
            cb_data.geometry.x = 0.1;
            image.main(cb_data,
                       left_images,
                       right_images,
                       shared,
                       span);

            // Assertions
            let actual, expected;
            actual = left_images.data["image"][0];
            expected = [1, 1, 1, 255,
                        1, 1, 1, 0,
                        1, 1, 1, 255,
                        1, 1, 1, 0];
            expect(actual).to.be.deep.equal(expected);
        });
    });
    describe("slide_image", function() {
        it("should turn off pixels given 1x4 image", function() {
            // Fixture
            let images = four_pixels([1, 4]);
            let mouse_x = 0.5;
            let previous_mouse_x = 0;
            let use_previous_mouse_x = false;

            // System under test
            image.slide_image("left",
                              images,
                              mouse_x,
                              previous_mouse_x,
                              use_previous_mouse_x)

            // Assertions
            let actual, expected;
            actual = images.data["image"][0];
            expected = [1, 1, 1, 255,
                        1, 1, 1, 255,
                        1, 1, 1, 0,
                        1, 1, 1, 0];
            expect(actual).to.be.deep.equal(expected);
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
