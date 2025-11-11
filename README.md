# MLZoomcamp 2025 midterm project: Classifying what seeds

## Introduction and problem description

Accurate and rapid seed identification is a critical pre-harvest task in agriculture, ensuring that growers are planting exactly the crops they intend. In this project, we apply classical ML techniques to classify wheat kernels based on physical measurements (e.g. length, width) derived from cost-effective X-ray images. We train and tune several classifiers, assess their performance and deploy the best performing model as a web-service.

The project utilizes the Seeds Dataset from the [UC Riverside Machine Learning repository](archive.ics.uci.edu/dataset/236/seeds), which comprises 210 observations across 7 unique physical feature measurements (e.g., asymmetry coefficient, length, and width) derived from X-ray images, accompanied by a variety label. See [below](exploring-the-dataset) for instructions on downloading the data set.

### Requirements

In order to run everything in this project you will need the following:

- Docker >= 28.5.0
- GNU `make` >=4.3
- `bash` >=5.0

To simplify various stages we use `make` commands. Run

```bash
make help
```

to see the available commands. In case `make` is unavailable, you can manually run the commands specified in the `Makefile`.

Note that this project has been tested in a Linux environment and should also run fine on MacOS. For Windows, we recommend running it in [WSL](https://learn.microsoft.com/en-us/windows/wsl/about).

## Setting up for development

To set up the `python` environment and all dependencies run

```bash
make setup_env
```

This will install `uv` (if not present) and install all dependecies needed

## Exporing the dataset

To download the dataset, please run

```bash
make get_dataset
```

You can find the EDA notebook [here](./notebook_seeds.ipynb). This notebook contains:

- details about the dataset, its characteristics
- training and tuning of several classification models and assessing their performance
  - Discriminative:
    - Logistic regression
    - k-Nearest Neighbors
    - (Kernel) Support Vector Machines (kSVM)
    - Random Forest Classifier
    - [XGBoost](https://xgboost.readthedocs.io/en/stable/)
    - [CatBoost](https://github.com/catboost/catboost)
    - [LightGBM](https://github.com/microsoft/LightGBM)

  - Generative:
    - Linear/Quadratic Discriminant Analysis
    - Naive Bayes
- picking a final model to use. We find that LDA is the best model.
See the notebook for more details.

## Training the production model

To train a model in production, we use a separate training script, `train.py`. To run the training do

```bash
make train
```

This will produce the model file `seeds_classifier.pkl` with the trained model as well as display the performance of the production model on the test such which was not used for tuning or training the model.

## Building and deploying a prediction service

We create a FastAPI-based prediction service that runs inside a Docker container. To build the container run

```bash
make build_prediction_service
```

This will create an image `seeds_classifier` which will contain the trained model and the web-based prediction service

To launch the prediction service run

```bash
make serve_predictions
```

To test with a sample payload, run

```bash
make test_prediction_service
```

This command runs the script `test_prediction_service.py` with a payload describing the physical characteristics of a single wheat sample,  and outputs the wheat variety, as well as the probability of belonging to that variety.  Feel free to play around with the numbers to see how the predictions change.

## Cleaning up

To stop the prediction service run

```bash
make shutdown_prediction_service
```

To remove the service Docker container entirely, do

```bash
make remove_prediction_container
```

Finally to get rid of the Python virtual environment, run

```bash
make cleanup_venv
```
