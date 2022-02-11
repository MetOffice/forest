let openId = function(id, width) {
    if (typeof width === "undefined") {
        width = "400px";
    }
    document.getElementById(id).style.width = width;
}
let closeId = function(id) {
    document.getElementById(id).style.width = "0";
}

/* Switch to/from "modal" context */
function openModal() {
    let els = {
        controls: document.getElementById('controls-container'),
        modal: document.getElementById('modal-container')
    }
    if (els.controls.classList.contains('display-block')) {
        els.controls.classList.remove('display-block')
        els.controls.classList.add('display-none')
        els.modal.classList.remove('display-none')
        els.modal.classList.add('display-block')
    }
}
function closeModal() {
    let els = {
        controls: document.getElementById('controls-container'),
        modal: document.getElementById('modal-container')
    }
    if (els.modal.classList.contains('display-block')) {
        els.modal.classList.remove('display-block')
        els.modal.classList.add('display-none')
        els.controls.classList.remove('display-none')
        els.controls.classList.add('display-block')
    }
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
