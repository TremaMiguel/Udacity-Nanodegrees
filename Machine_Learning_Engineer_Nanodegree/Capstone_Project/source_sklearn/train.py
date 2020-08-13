import argparse
import os
import json
import pandas as pd
import joblib

from sklearn.linear_model import BayesianRidge


# Provided model load function (Taken from Case Studies Project)
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")

    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")

    return model


if __name__ == "__main__":

    # Initialize an ArgumentParser
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument(
        "--output-data-dir", type=str, default=os.environ["SM_OUTPUT_DATA_DIR"]
    )
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAIN"])

    # Add model parameters
    parser.add_argument(
        "-p",
        "--mdl_params",
        type=str,
        default="{}",
        required=True,
        help='Classifier Params (default: "")',
    )
    # NOTE: A dictionary should be passed as a string. See more here:
    # https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python

    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(
        os.path.join(training_dir, "train.csv"), header=None, names=None
    )

    # Labels are in the first column
    train_y = train_data.iloc[:, 0]
    train_x = train_data.iloc[:, 1:]

    # Load params
    mdl_params = json.loads(args.mdl_params)

    # Define model
    model = BayesianRidge(**mdl_params)

    # Train Model
    model.fit(train_x, train_y)

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))
