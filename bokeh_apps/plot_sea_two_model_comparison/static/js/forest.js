let Forest = (function() {
    // Plugin to customize layout
    let timerID;
    let bokehLoaded = function() {
        return Bokeh.documents.length > 0;
    };
    let main = function() {
        if (bokehLoaded()) {
            clearInterval(timerID);
        } else {
            return;
        }

        // Bokeh has finished loading
        console.log("forest.js: Bokeh loaded");
    };
    timerID = setInterval(main, 50);
    return {
        main: main
    };
})();
