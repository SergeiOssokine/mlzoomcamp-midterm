import logging
import os
from pickle import dump

import numpy as np
import pandas as pd
import typer
from rich.logging import RichHandler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from typing_extensions import Annotated

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(rich_tracebacks=True, markup=True))
logger.setLevel("INFO")


def load_data(filename: str) -> pd.DataFrame:
    """Load the seeds data

    Args:
        filename (str): Name containing the raw data

    Returns:
        pd.DataFrame: Dataframe with raw data
    """
    raw_data = np.genfromtxt(filename)
    cols = [
        "area",
        "perimeter",
        "compactness",
        "length_kernel",
        "width_kernel",
        "asymmetry_coeff",
        "length_kernel_groove",
        "variety",
    ]
    df = pd.DataFrame(raw_data, columns=cols)
    return df


random_state = 27
np.random.seed(seed=random_state)


def split_data(
    df: pd.DataFrame, test_frac: float = 0.2, target_var="variety", random_state=27
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Split the data set into training and testing with test_frac samples in the testing set.
    Also separates the dependent and indepdent variables.

    Args:
        df (pd.DataFrame): The raw dataset
        test_frac (float, optional): Fraction of samples to reserve for testin. Defaults to 0.2.
        target_var (str, optional): The name of the dependent variable. Defaults to "variety".
        random_state (int, optional): Random seed to use. Defaults to 27.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]: df_train, df_test, y_train, y_test
    """
    df_train, df_test = train_test_split(
        df, test_size=test_frac, random_state=random_state
    )
    y_train = df_train[target_var].values.astype("int")
    y_test = df_test[target_var].values.astype("int")

    df_train = df_train.drop(columns=[target_var])
    df_test = df_test.drop(columns=[target_var])

    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_test, y_train, y_test


def prepare_data(df: pd.DataFrame, test_frac: float = 0.2):
    # Variety is a numerical label, make sure it's int, not float. Also make labels be 0-based
    df["variety"] = df["variety"].astype(int) - 1
    test_frac = 0.2

    # Perform train/test split.
    logger.info(
        f"Peforming train test split with testing fraction of {test_frac * 100}%"
    )
    df_train, df_test, y_train, y_test = split_data(
        df, random_state=random_state, test_frac=test_frac
    )

    logger.info(
        f"Done. Train test has {len(df_train)} samples, test set has {len(df_test)} samples"
    )
    return df_train, df_test, y_train, y_test


def train_model(df_train: pd.DataFrame, y_train: np.ndarray):
    clf = LinearDiscriminantAnalysis()
    clf.fit(df_train, y_train)
    return clf


def test_model(fitted_model, df_test: pd.DataFrame, y_test: np.ndarray):
    logger.info("Assessing model performance on the test set")
    y_pred = fitted_model.predict(df_test)
    print(classification_report(y_test, y_pred))


def main(
    filename: Annotated[
        str, typer.Option(help="The name of the file with the seeds data")
    ] = os.path.join("data", "seeds_dataset.txt"),
    output_name: Annotated[
        str, typer.Option(help="File to store model in")
    ] = "seeds_classifier.pkl",
):
    logger.info("[bold]Training production model[/bold]")
    logger.info(f"Loading data from {filename}")
    df = load_data(filename)
    logger.info("Done")
    logger.info("Preparing data")
    df_train, df_test, y_train, y_test = prepare_data(df)
    # Train the model
    logger.info("Training the Linear Discriminant Analysis model")
    fitted_model = train_model(df_train, y_train)
    logger.info("Done")

    test_model(fitted_model, df_test, y_test)

    # Save the model
    logger.info(f"Saving the fitted model to {output_name}")
    with open(output_name, "wb") as fw:
        dump(fitted_model, fw, protocol=5)
    logger.info("All done")


if __name__ == "__main__":
    typer.run(main)
