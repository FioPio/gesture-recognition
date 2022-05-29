This project is based on (https://github.com/kinivi/hand-gesture-recognition-mediapipe) (https://github.com/kinivi/hand-gesture-recognition-mediapipe .

# Install the environment
First you have to clone and get into the repo directory:
```bash
git clone https://github.com/FioPio/gesture-recognition.git
cd gesture-recognition
```
To use the same environment used in this project, you may just install the conda env that I provide by:

```bash
conda env create -f environment.yml
```
then the environment may be activated and deactivated as:

```bash
conda activate gesture
```

and deactivated as:

```bash
conda deactivate
```


# Get started
To adquire data and check that everything is working you may run the `getData` script:

```bash
python getData.py
```


You may record postures by pressing a key number between 0 and 9 (should be done in order) meanwhile you have the pose with your hand. The data is saved in `training_data`, so this folder should be created. Finally, the train may be run by just:

```bash
python train.py 
``` 

which will train the model and save it under the model name specified in the code in the `model` folder.

# Running the predictor

To run the predictor :

```bash
python predict.py
```

This will allow you to control the windows on the screen by gestures of the right hand that you have previously trained.