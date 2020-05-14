(function(){function r(e,n,t){function o(i,f){if(!n[i]){if(!e[i]){var c="function"==typeof require&&require;if(!f&&c)return c(i,!0);if(u)return u(i,!0);var a=new Error("Cannot find module '"+i+"'");throw a.code="MODULE_NOT_FOUND",a}var p=n[i]={exports:{}};e[i][0].call(p.exports,function(r){var n=e[i][1][r];return o(n||r)},p,p.exports,r,e,n,t)}return n[i].exports}for(var u="function"==typeof require&&require,i=0;i<t.length;i++)o(t[i]);return o}return r})()({1:[function(require,module,exports){
const helloWorld = () => "Hello, World!"


// Populate variable options from dataset value
const link_selects = function(dataset_select, variable_select, source) {
    let label = dataset_select.value;
    if (label !== "") {
        let index = source.data['datasets'].indexOf(label)
        let defaults = ["Please specify"];
        variable_select.options = defaults.concat(
            source.data['variables'][index]);
    }
}


module.exports = {
    helloWorld,
    link_selects
}

},{}],2:[function(require,module,exports){
window.forest = require('./forest')


},{"./forest":1}]},{},[2]);
