let Forest = (function() {
    // Plugin to customize layout
    let timerID;
    let bokehLoaded = function() {
        return Bokeh.documents.length > 0;
    };
    let fullScreen = function(figure) {
        figure.width = window.innerWidth;
        figure.height = window.innerHeight;
    };
    let main = function() {
        if (bokehLoaded()) {
            clearInterval(timerID);
        } else {
            return;
        }
        let roots = Bokeh.documents[0].roots();
        let figures = roots.filter(root => root.name === "figure");
        figures.map(fullScreen);
        window.dispatchEvent(new Event('resize'));
        window.addEventListener('resize', function() {
            figures.map(fullScreen);
        });
    };
    timerID = setInterval(main, 50);
    return {
        main: main
    };
})();
