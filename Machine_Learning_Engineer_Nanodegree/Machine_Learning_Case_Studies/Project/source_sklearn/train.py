from __future__ import print_function

import argparse
import os
import json 
import pandas as pd
import joblib

#from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

## TODO: Import any additional libraries you need to define a model


# Provided model load function
def model_fn(model_dir):
    """Load model from the model_dir. This is the same model that is saved
    in the main if statement.
    """
    print("Loading model.")
    
    # load using joblib
    model = joblib.load(os.path.join(model_dir, "model.joblib"))
    print("Done loading model.")
    
    return model


## TODO: Complete the main code
if __name__ == '__main__':
    
    # All of the model parameters and training parameters are sent as arguments
    # when this script is executed, during a training job
    
    # Here we set up an argument parser to easily access the parameters
    parser = argparse.ArgumentParser()

    # SageMaker parameters, like the directories for training data and saving models; set automatically
    # Do not need to change
    parser.add_argument('--output-data-dir', type=str, default=os.environ['SM_OUTPUT_DATA_DIR'])
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--data-dir', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    
    ## TODO: Add any additional arguments that you will need to pass into your model
    parser.add_argument('-c','--classifier', type=str, default='AdaBoost', required=True, 
                        help='Classifier Model (default: AdaBoost)')
    parser.add_argument('-p', '--classifier_params', type=str, default='{}', required=True,
                        help='Classifier Params (default: "")')
    # NOTE: A dictionary should be passed as a string. See more here: 
    #https://stackoverflow.com/questions/18608812/accepting-a-dictionary-as-an-argument-with-argparse-and-python
                        
    
    # args holds all passed-in arguments
    args = parser.parse_args()

    # Read in csv training file
    training_dir = args.data_dir
    train_data = pd.read_csv(os.path.join(training_dir, "train.csv"), header=None, names=None)

    # Labels are in the first column
    train_y = train_data.iloc[:,0]
    train_x = train_data.iloc[:,1:]
    
    
    ## --- Your code here --- ##
    

    ## TODO: Define a model 
    mdl = args.classifier
    mdl_args = json.loads(args.classifier_params)
    
    if mdl == 'GaussianNB':
        model = GaussianNB(**mdl_args)
    elif mdl == 'QDA':
        model = QuadraticDiscriminantAnalysis(**mdl_args)
    else: 
        model = AdaBoostClassifier(**mdl_args)
    
    
    ## TODO: Train the model
    model.fit(train_x, train_y)
    
    
    ## --- End of your code  --- ##
    

    # Save the trained model
    joblib.dump(model, os.path.join(args.model_dir, "model.joblib"))