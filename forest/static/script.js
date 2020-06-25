let openId = function(id, width) {
    if (typeof width === "undefined") {
        width = "400px";
    }
    document.getElementById(id).style.width = width;
}
let closeId = function(id) {
    document.getElementById(id).style.width = "0";
}
