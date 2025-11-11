# MLZoomcamp 2025 midterm project: Classifying what seeds

## Introduction and problem description

The use of machine learning techniques in agriculture has been growing steadily in recent years.

The dataset was obtained from UC Riverside Machine Learning repository [here](archive.ics.uci.edu/dataset/236/seeds). See [below](exploring-the-dataset) for instructions on downloading it.

In order to run everything in this project you will need the following:

- Docker >= 28.5.0
- GNU `make` >=4.3

To simplify various stages we make use of `make` commands. Run

```bash
make help
```

to see the available commands. In case `make` is unavailable, you can manually run the commands specified in the `Makefile`.

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
- training and tuning many different classification models and assessing their performance
- picking a final model to use
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
