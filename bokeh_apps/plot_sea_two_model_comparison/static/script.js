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

// CSS reset to control BokehJS defaults
let ready = function(callback, opts) {
    // setInterval/clearInterval
    let intervalID;
    let setUp = function() {
        if (Bokeh.documents[0].roots().length === opts.roots) {
            // Bokeh not loaded continue
            return;
        }
        callback();
        clearInterval(intervalID);
    };
    intervalID = setInterval(setUp, 50);
};
let resetCSS = function() {
    let els = document.getElementsByClassName("forest-nav");
    for (let i=0; i<els.length; i++) {
        let el = els[i];
        el.style.height = "auto";
    }
};
window.addEventListener('load',
    ready(resetCSS, {roots: 3})
);
