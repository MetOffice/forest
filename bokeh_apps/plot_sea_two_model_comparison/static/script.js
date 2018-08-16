"use strict";
let rootIndex = 1;
let resize = function() {
    let figure = Bokeh.documents[0].roots()[rootIndex];
    figure.width = window.innerWidth;
    figure.height = window.innerHeight;
};
window.addEventListener('resize', resize);
let load = function() {
    // setInterval/clearInterval solution
    let intervalID;
    let setUp = function() {
        if (Bokeh.documents[0].roots().length === 0) {
            // continue
            return;
        }
        let figure = Bokeh.documents[0].roots()[rootIndex];
        if (figure.width === window.innerWidth) {
            clearInterval(intervalID);
        } else {
            figure.width = window.innerWidth;
            figure.height = window.innerHeight;
            window.dispatchEvent(new Event('resize'));
        }
    };
    intervalID = setInterval(setUp, 50);
};
window.addEventListener('load', load);

