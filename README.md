# Chest Tube and Pneumothorax Classification in Chest X-rays using Deep Learning

### Setup
1. Clone this repository
2. Install [poetry](https://python-poetry.org/docs/#installation)
3. Install dependencies: `poetry install`
4. Add the data to the `chestxray-data` folder
5. Set the required environment variables to specify the paths to the data and cache directory:
   ```
   export BACHELOR_THESIS_DATA_DIR="/path/to/data/directory"
   export BACHELOR_THESIS_CACHE_DIR="/path/to/cache/directory"
   ```

### Chest tube classification training:

`python src/ptx_classification/chest_tube_classification/train_ct_candid.py`

### Pneumothorax classification training:

With CANDID-PTX dataset:

`python src/ptx_classification/ptx/train_ptx_candid.py`

With CheXpert dataset:

`python src/ptx_classification/ptx/train_chexpert.py`

With ChestX-ray14 dataset:

`python src/ptx_classification/ptx/train_chestxray14.py`
