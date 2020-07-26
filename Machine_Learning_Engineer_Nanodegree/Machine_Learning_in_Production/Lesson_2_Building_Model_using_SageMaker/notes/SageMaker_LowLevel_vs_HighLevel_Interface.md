# Sage Maker Low Level and High Level Interface 

We're going to see examples of how to configure a model through the **Low Level** and **High Level** Interface of AWS SageMaker. 


## <p><span style="color:blue">The High Level Interface Example</span><p> 

We initially set the Sage Maker session and get the role SageMaker will be using to access the S3 bucket. 

```
import sagemaker
from sagemaker import get_execution_role
from sagemaker.predictor import csv_serializer

# Set a SageMaker session 
session = SageMaker.session() 

# Get the execution role 
role = get_execution_role()
```

### Step 1. Save Data Locally 

The estimator of SageMaker does not get `headers` and `index`, thus we set this parameter to `False` when saving the dataframe. 

```
import os 

# Create a directory to save the data 
data_dir = '../data/boston'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# Save to local directory 
test.to_csv(os.path.join(data_dir, 'test.csv'), 
            header=False,
            index=False)
```

### Step 2. Upload data to S3 

Notice that as we're using the High Level Interface, we will be storing the data in the default S3 bucket. 

```
prefix = 'boston-xgboost-HL'

test_location = session.upload_data(os.path.join(data_dir, 'test.csv'),key_prefix=prefix)
val_location = session.upload_data(os.path.join(data_dir, 'validation.csv'), key_prefix=prefix)
train_location = session.upload_data(os.path.join(data_dir, 'train.csv'), key_prefix=prefix)
```

### Step 3. Train a Model 

To use a prebuilt SageMaker model, we need to provide the Docker container image, we can do this through the function `get_image_uri`, which gets as input the region where we're performing the computation and the name of the algorithm we want. For this example, we will be using **XGBoost** as model, therefore, we can set some hyperparameters. 

```
from sagemaker.amazon.amazon_estimator import get_image_uri

# Set container 
container = get_image_uri(session.boto_region_name, 'xgboost')

# Set Estimator 
xgb = sagemaker.estimator.Estimator(container, # The image name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance to use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

# Set model hyperparameters 
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

# Set S3 location of train-validation data 
s3_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_val = sagemaker.s3_input(s3_data=val_location, content_type='csv')

# Fit estimator
xgb.fit({'train': s3_train, 'validation': s3_val})
```

Notice that we need to specify the type of the data, in this case it is `.csv`. 


### Step 4: Test the model 

One can use [SageMaker Transformer](https://docs.aws.amazon.com/sagemaker/latest/dg/ex1-batch-transform.html) to test the model in a batch way. 

```
# Set transformer 
xgb_transformer = xgb.transformer(instance_count = 1, instance_type = 'ml.m4.xlarge')

# Set transform Job 
xgb_transformer.transform(test_location, content_type='text/csv', split_type='Line')

# Wait for results 
xgb_transformer.wait()
```

For Full details, you can check out the following [notebook](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/Boston%20Housing%20-%20XGBoost%20(Batch%20Transform)%20-%20High%20Level.ipynb). 


## <p><span style="color:blue">The Low Level Interface Example</span><p> 

The Low Level Interface allows you to take more flexible decisions regarding the model and the transformer. The previous steps of saving and uploading data to S3 are similar, thus we're going to focus only on training and testing the model. 

### Step 1. Set up parameters for SageMaker Job 

Before we set the hyperparameters and the estimator through functions (High Level Interface), now we're going to store this configurations in a dictionary to pass to a SageMaker `trainingjob`. 

```
# Set a container
container = get_image_uri(session.boto_region_name, 'xgboost')

# Create a params dict 
training_params = {}

# Set a Role to execute tasks
training_params['RoleArn'] = role

# Specify the model 
training_params['AlgorithmSpecification'] = {
    "TrainingImage": container,
    "TrainingInputMode": "File"
}

# Location to the model artifacts
training_params['OutputDataConfig'] = {
    "S3OutputPath": "s3://" + session.default_bucket() + "/" + prefix + "/output"
}

# Set a compute instance 
training_params['ResourceConfig'] = {
    "InstanceCount": 1,
    "InstanceType": "ml.m4.xlarge",
    "VolumeSizeInGB": 5
}

# Set a stopping condition for the machine     
training_params['StoppingCondition'] = {
    "MaxRuntimeInSeconds": 86400
}

# Set model hyperparameters
training_params['HyperParameters'] = {
    "max_depth": "5",
    "eta": "0.2",
    "gamma": "4",
    "min_child_weight": "6",
    "subsample": "0.8",
    "objective": "reg:linear",
    "early_stopping_rounds": "10",
    "num_round": "200"
}

# Location of data for the model 
training_params['InputDataConfig'] = [
    {
        "ChannelName": "train",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": train_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    },
    {
        "ChannelName": "validation",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": val_location,
                "S3DataDistributionType": "FullyReplicated"
            }
        },
        "ContentType": "csv",
        "CompressionType": "None"
    }
]
```

### Step 2. Create a SageMaker Job 

```
# Set Name for SageMaker Training Job
training_params['TrainingJobName'] = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Create and execute the training job
training_job = session.sagemaker_client.create_training_job(**training_params)

# Wait for job to finish 
session.logs_for_job(training_job_name, wait=True)
```



### Step 3. Create a Model 

Now that the model has been trained, we can retrieve those artifacts to create a model for making inference. To do this, we first retrieve the artifacts from the model outputed by the Training Job and specify these artifacts to the SageMaker model. 

``` 
# Retrieve info of the Training Job 
training_job_info = session.sagemaker_client.describe_training_job(TrainingJobName=training_job_name)

# Retrieve artifacts from Training Job 
model_artifacts = training_job_info['ModelArtifacts']['S3ModelArtifacts']

# Set a name to the model 
model_name = training_job_name + "-model"

# Container to be used for inference and where it should retrieve the model artifacts from.
primary_container = {
    "Image": container,
    "ModelDataUrl": model_artifacts
}

# Set SageMaker model
model_info = session.sagemaker_client.create_model(
                                ModelName = model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = primary_container)
```



### Step 4. Set a Batch Transform Job 

```
# Set Name for SageMaker Transform Job 
transform_job_name = 'boston-xgboost-batch-transform-' + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Set params for Transform Job 
transform_request = \
{
    "TransformJobName": transform_job_name,
    
    # This is the name of the model that we created earlier
    "ModelName": model_name,
    
    # Set how many compute instances should be used at once
    "MaxConcurrentTransforms": 1,
    
    # Size of individual requests to the model.
    "MaxPayloadInMB": 6,
    
    # Each of the chunks that we send to the endpoint should contain multiple samples of our input data.
    "BatchStrategy": "MultiRecord",
    
    # Where the output data should be stored
    "TransformOutput": {
        "S3OutputPath": "s3://{}/{}/batch-bransform/".format(session.default_bucket(),prefix)
    },
    
    # On which S3 bucket is data stored  
    "TransformInput": {
        "ContentType": "text/csv",
        "SplitType": "Line",
        "DataSource": {
            "S3DataSource": {
                "S3DataType": "S3Prefix",
                "S3Uri": test_location,
            }
        }
    },
    
    # Set a compute instance 
    "TransformResources": {
            "InstanceType": "ml.m4.xlarge",
            "InstanceCount": 1
    }
}

# Set up and execute Transform Job  
transform_response = session.sagemaker_client.create_transform_job(**transform_request)

# Wait for Tranform Job to finish 
transform_desc = session.wait_for_transform_job(transform_job_name)
```
