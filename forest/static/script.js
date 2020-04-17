let openId = function(id) {
    document.getElementById(id).style.width = "400px";
}
let closeId = function(id) {
    document.getElementById(id).style.width = "0";
}

let forest = (function() {
    let ns = {};

    // Populate variable options from dataset value
    ns.link_selects = function(dataset_select, variable_select, source) {
        let label = dataset_select.value;
        if (label !== "") {
            let index = source.data['datasets'].indexOf(label)
            let defaults = ["Please specify"];
            variable_select.options = defaults.concat(
                source.data['variables'][index]);
        }
    }

    return ns;
})();
