let openId = function(id, width) {
    if (typeof width === "undefined") {
        width = "400px";
    }
    document.getElementById(id).style.width = width;
}
let closeId = function(id) {
    document.getElementById(id).style.width = "0";
}

// Cross-browser full screen
function openFullscreen(el) {
    if (el.requestFullscreen) {
        el.requestFullscreen()
    } else if (el.mozRequestFullScreen) {
        /* Mozilla */
        el.mozRequestFullScreen()
    } else if (el.webkitRequestFullscreen) {
        /* Chrome, Safari, Opera */
        el.webkitRequestFullscreen()
    } else if (el.msRequestFullscreen) {
        /* IE/Edge */
        el.msRequestFullscreen()
    }
}


/* Close fullscreen */
function closeFullscreen() {
  if (document.exitFullscreen) {
    document.exitFullscreen();
  } else if (document.mozCancelFullScreen) { /* Firefox */
    document.mozCancelFullScreen();
  } else if (document.webkitExitFullscreen) { /* Chrome, Safari and Opera */
    document.webkitExitFullscreen();
  } else if (document.msExitFullscreen) { /* IE/Edge */
    document.msExitFullscreen();
  }
}

function getFullscreenElement() {
    return document.fullscreenElement || /* Standard syntax */
        document.webkitFullscreenElement || /* Chrome, Safari and Opera syntax */
        document.mozFullScreenElement ||/* Firefox syntax */
        document.msFullscreenElement /* IE/Edge syntax */
}
