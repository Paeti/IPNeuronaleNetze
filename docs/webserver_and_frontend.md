# Documentation for the webserver and frontend

## Webserver

## Frontend
* The ratio of the Canvas has to be the proportions of the Webcam, because otherwise the picture it takes will be stretched or compressed
```
<canvas id="canvas" width="400" height="300" hidden></canvas>
```
* The function **sendImage** does following things:
  * takes snapshot of the video from the webcam and draws it into a canvas
  * this canvas will then be converted to base64
  * the base64 string will now be send to the server in jsonFormat via AJAX
  * if it gets a response from the server it parses the JSON and writes  the attributes into the frontend

* The function **sendForm** has following things it does:
  * it gets the values of the form in the frontend
  * and will send those to the server with ajax aswell

