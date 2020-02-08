let openSideNav = function() {
    document.getElementById("sidenav").style.width = "400px";
    document.getElementById("sidenav").style.borderLeftWidth = "1px";
    document.getElementById("main").style.marginLeft = "401px";
}
let closeSideNav = function() {
    document.getElementById("sidenav").style.width = "0";
    document.getElementById("sidenav").style.borderLeftWidth = "0";
    document.getElementById("main").style.marginLeft = "0";
}
