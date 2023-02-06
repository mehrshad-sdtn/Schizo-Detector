# Dog-Breed-Classifier
A Deep Learning model for dog breed classification using Transfer Learning
[Stanford dog dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/) was used for this project with <br>
[Xception](https://arxiv.org/abs/1610.02357) CNN architecture, Xception pre-trained weights were used, and the final layer (FC) was trained

#### Xception Architecture
<img src="https://github.com/mehrshad-sdtn/Dog-Breed-Classifier/blob/master/images/Xception.png" alt="xception" style="width:600px;"/>


### Results
**75.2% accuracy** was achieved on the validation data

**[notebook](https://github.com/mehrshad-sdtn/whatdogami/tree/master/notebooks) for model training**

### Deployment
the project is deployed as a web application via Heroku and with Flask framework and could be found at [this address](https://whatdogami.herokuapp.com/)
