# Documentation for the webserver and frontend

## Webserver
Before you start the server, you have to install all the dependencies, that are requierd

* To start the server go into the web directory and type in:
    ```
    set FLASK_APP=server.py 
    flask run
    ```

* use the shown url: **localhost:5000**, and as you let your age and gender get predicted, you get your predictions shown in the front end. The taken pictures is saved in /data/production_img, with a unique id as png. 
    --> Example: 5_.png

* you can now adjust your age and gender in the front end, and if you tick the save box, the new parameters will be written into the pictures name --> 
    Example: 5_45_M.png
    Otherwise the picture will be deleted!

### Technically
* localhost:5000/ --> renders the index page,
* the front end sends per ajax the taken picture back to: /prediciton,
* /prediction takes care of saving the picture and gets the age and gender parameters,
* the image gets preprocessed and converted to an array and is sent to the models, the models send the parameters back,
* the parameters are sent back to the front end, 
* a second ajax in the front end takes care of sending back the new parameters,
* and the /save route either deletes the picture or adds the parameters to the name.

## Frontend
* **The ratio of the Canvas has to be the proportions of the Webcam**, because otherwise the picture it takes will be stretched or compressed
```
<canvas id="canvas" width="400" height="300" hidden></canvas>
```
* If the Button *Lass dich schätzen* is clicked the UI changes and the function **sendImage** will be called

  * The function **sendImage** does following things:
    * takes snapshot of the video from the webcam and draws it into a canvas
    * this canvas will then be converted to base64
    * the base64 string will now be send to the server in jsonFormat via AJAX
    * if it gets a response from the server it parses the JSON and writes  the attributes into the frontend

* If the Button *Neuer Versuch* is clicked the UI changes and the function **sendForm** will be called

  * The function **sendForm** has following things it does:
    * it gets the values of the form in the frontend
    * and will send those to the server with ajax aswell


  

