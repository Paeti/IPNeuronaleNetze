const webcam = document.getElementById('webcam');
const canvas = document.getElementById('canvas');
const user = document.getElementById('userForm');
const loader = document.getElementById('heartLoader');
const context = canvas.getContext('2d');
const biggerText = document.getElementById('issaBiggerText');
const smallerText = document.getElementById('issaSmallerText');
const constraints = {
  video: true,
};

//Enables webcam
navigator.mediaDevices.getUserMedia(constraints)
  .then((stream) => {
    webcam.srcObject = stream;
  });


// Checks if "^" Key is pressed
document.addEventListener("keydown", function (e) {
  if (e.keyCode == 192) {
    // Call the toggle function
    toggleFullScreen();
  } else if (e.keyCode == 13) {
    // Draw the video frame to the canvas and make second stage visible.
    context.drawImage(webcam, 0, 0, canvas.width, canvas.height);
    //send(canvas);
    user.style.opacity = 0;
    user.style.transitionDuration = "1s";
    loader.style.opacity = 1;
    loader.style.transitionDuration = "1s";
    smallerText.style.opacity = 0;
    smallerText.style.transitionDuration = "1s";
    crypticWow("Ihr Bild wird nun verarbeitet");
  } else if (e.keyCode == 77) {
    crypticWow("Sie sind 54 Jahre alt und weiblich");
      smallerText.innerHTML = "korrigieren Sie ihre Schätzung und drücken Sie dann auf das Herz";
    smallerText.style.transitionDelay = "1s";
    smallerText.style.opacity = 1;
    smallerText.style.transitionDuration = "2s";

  }
}, false);

function send(canvas) {
  var dataURL = canvas.toDataURL();
  $.ajax({
    type: "POST",
    url: "", //TODO: insert filename which gets the image via post
    data: {
       imgBase64: dataURL
    }
  }).done(function(o) {
    console.log('saved');
  });
}

function crypticWow(text) {
  setTimeout(function () {
    var logoRandom = '';
    var possible = "-+*/|}{[]~:;?/.><=+-_)(*&^%$#@!)}...";
    for (var i = 0; i < text.length + 1; i++) {
      logoRandom = text.substr(0, i);
      for (var j = i; j < text.length; j++) {
        logoRandom += possible.charAt(Math.floor(Math.random() * possible.length));
      }
      generateRandomTitle(i, logoRandom);
      logoRandom = '';
    }
    function generateRandomTitle(i, logoRandom) {
      setTimeout(function () {
        biggerText.innerHTML = logoRandom;
      }, i * 50);
    }
  }, 100);
};
// Enables Fullscreenmode
function toggleFullScreen() {
  if (!document.fullscreenElement) {
    var video = document.getElementById("webcam");
    if (video.requestFullscreen) {
      video.requestFullscreen();
    } else if (video.mozRequestFullScreen) {
      video.mozRequestFullScreen();
    } else if (video.webkitRequestFullscreen) {
      video.webkitRequestFullscreen();
    } else if (video.msRequestFullscreen) {
      video.msRequestFullscreen();
    }
  } else {
    if (document.exitFullscreen) {
      document.exitFullscreen();
    }
  }
}
