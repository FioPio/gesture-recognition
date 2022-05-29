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