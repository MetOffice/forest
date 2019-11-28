const { watch, src, dest, parallel } = require('gulp'), less = require('gulp-less'), minifyCSS = require('gulp-csso'), concat = require('gulp-concat'), minify = require("gulp-minify");

function buildCss() {
  return src('less/*.less')
    .pipe(less())
    .pipe(concat('forest.css'))	
    .pipe(minifyCSS())
    .pipe(dest('static/css'))
}

function buildJs() {
    return src('js/forest.js', { allowEmpty: true }) 
        .pipe(minify({noSource: true}))
        .pipe(dest('static/js'))
}

function watchAll(){
  watch('less/*.less', buildCss);
  watch('js/*.js', buildJs);
}

exports.watchAll = watchAll;
exports.default = parallel(buildCss, buildJs);