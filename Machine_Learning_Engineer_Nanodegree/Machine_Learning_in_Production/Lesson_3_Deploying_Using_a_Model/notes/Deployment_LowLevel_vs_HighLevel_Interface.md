# Deployment Low Level and High Level Interface 

Similarly as we've seen in the notes `SageMaker_LowLevel_vs_HighLevel_Interface`. We'll be seeing two different ways of deploying a 
model with AWS stack. Once the data is uploaded to S3, check those steps on prior the notebook, we can construct the model and train it. 

## <p><span style="color:blue">The High Level Interface Example</span><p>


### Step 1. Train the model 

```
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# Set Session and role 
session = SageMaker.session()
role = get_execution_role()

# Set the training container.
container = get_image_uri(session.boto_region_name, 'xgboost')

# Set the estimator object.
xgb = sagemaker.estimator.Estimator(container, # The name of the training container
                                    role,      # The IAM role to use (our current role in this case)
                                    train_instance_count=1, # The number of instances to use for training
                                    train_instance_type='ml.m4.xlarge', # The type of instance to use for training
                                    output_path='s3://{}/{}/output'.format(session.default_bucket(), prefix),
                                                                        # Where to save the output (the model artifacts)
                                    sagemaker_session=session) # The current SageMaker session

# Set Model hyperparameters 
xgb.set_hyperparameters(max_depth=5,
                        eta=0.2,
                        gamma=4,
                        min_child_weight=6,
                        subsample=0.8,
                        objective='reg:linear',
                        early_stopping_rounds=10,
                        num_round=200)

# In this example train_location and val_location are files stored locally, for example, data/train.csv. 
s3_input_train = sagemaker.s3_input(s3_data=train_location, content_type='csv')
s3_input_validation = sagemaker.s3_input(s3_data=val_location, content_type='csv')

# Train the model 
xgb.fit({'train': s3_input_train, 'validation': s3_input_validation})
```


### Step 2. Deploy the model 

As we're using the **High Level** Interface approach, this will require a single line of code. It is important to know that behind the 
curtains SageMaker is launching a compute instance that will run until it is shut down. That is, the cost of a deployed endpoint depends on how long it has been running for.

```
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')
```

### Step 3. Send data to the Endpoint 

```
# Specify the type of data the endpoint is expecting 
xgb_predictor.content_type = 'text/csv'
xgb_predictor.serializer = csv_serializer

# Call the endpoint on test data 
Y_pred = xgb_predictor.predict(X_test.values).decode('utf-8')
Y_pred = np.fromstring(Y_pred, sep=',') # Y_pred is a csv file
```

Finally, remember to delete the endpoint to avoid surprising charges. 

```
xgb_predictor.delete_endpoint()
```

## <p><span style="color:blue">The Low Level Interface Example</span><p>

We saw before on the `SageMaker_LowLevel_vs_HighLevel_Interface` notes how to build a model through the Low Level Interface. Thus, as our objective is now the High Level Interface, we're going to focus on Low Level Interface approach for the deployment. Again, the steps to build the model are similar as before. 

### Step 1. Create and Deploy an Endpoint 

```
# Set a Name for the model 
model_name = "boston-xgboost-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime()) + "-model"

endpoint_config_name = "boston-xgboost-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Configure the endpoint 
endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": model_name,
                                "VariantName": "AllTraffic"
                            }])

# Set a Name for the endpoint
endpoint_name = "boston-xgboost-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Create an endpoint 
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = endpoint_config_name)

# Wait for endpoint to be created 
endpoint_dec = session.wait_for_endpoint(endpoint_name)
```


### Step 2. Send data to the endpoint 

Notice that the input to the `body` of the call is the data from which we would to receive predictions. 

```
# Invoke the endpoint 
response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = data)

# Deserialize the response 
result = response['Body'].read().decode("utf-8")
Y_pred = np.fromstring(result, sep=',')

# Delete the endpoint when you're finished 
session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)
```

## <p><span style="color:blue">Exposing the Endpoint through Lambda</span><p>

Currently the only way we can access the endpoint to send it data is using the SageMaker API. But, this endpoint requires the entity accessing it have the correct permissions. We can use AWS Lambda as backend (we will give it the permissions to send and receive data from the SageMaker Endpoint) to do all the data processing and API Gateway to expose this endpoint as a URL. This URL listens for data to be sent to it and pass it to the Lambda function. The architecture is illustrated in the image below 

![Infrastructure](Infrastructure_DM_6.png)

### Step 1. Set up a Lambda Function 

The code below should go on the Lambda Function created on AWS 

```
import boto3
import re

REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\()|(\))|(\[)|(\])")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

## Data Processing 

def review_to_words(review):
    words = REPLACE_NO_SPACE.sub("", review.lower())
    words = REPLACE_WITH_SPACE.sub(" ", words)
    return words

def bow_encoding(words, vocabulary):
    bow = [0] * len(vocabulary) # Start by setting the count for each word in the vocabulary to zero.
    for word in words.split():  # For each word in the string
        if word in vocabulary:  # If the word is one that occurs in the vocabulary, increase its count.
            bow[vocabulary[word]] += 1
    return bow

## Lambda Function 

def lambda_handler(event, context):

    vocab = "*** ACTUAL VOCABULARY GOES HERE ***"

    words = review_to_words(event['body'])
    bow = bow_encoding(words, vocab)

    # The SageMaker runtime is what allows us to invoke the endpoint that we've created.
    runtime = boto3.Session().client('sagemaker-runtime')

    # Now we use the SageMaker runtime to invoke our endpoint, sending the review we were given
    response = runtime.invoke_endpoint(EndpointName = '***ENDPOINT NAME HERE***',# The name of the endpoint we created
                                       ContentType = 'text/csv',                 # The data format that is expected
                                       Body = ','.join([str(val) for val in bow]).encode('utf-8')) # The actual review

    # The response is an HTTP response whose body contains the result of our inference
    result = response['Body'].read().decode('utf-8')

    # Round the result so that our web app only gets '1' or '0' as a response.
    result = round(float(result))

    return {
        'statusCode' : 200,
        'headers' : { 'Content-Type' : 'text/plain', 'Access-Control-Allow-Origin' : '*' },
        'body' : str(result)
    }
```

Notice that we can get `Endpoint Name` by calling `xgb_predictor.endpoint`. 


### Step 2. Set up a REST API

Following the steps on this [link](https://github.com/udacity/sagemaker-deployment/blob/master/Tutorials/IMDB%20Sentiment%20Analysis%20-%20XGBoost%20-%20Web%20App.ipynb) we can set a REST API through AWS API Gateway. 