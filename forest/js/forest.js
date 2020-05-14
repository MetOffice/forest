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
