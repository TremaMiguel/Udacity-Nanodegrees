# Updating an Endpoint 

Once an endpoint is deployed, it can be updated by passing a new configuration. We're going to illustrate this by first deploying 
two models, then updating this endpoint based on a single one. 

## Step 1. Constructing the models 

First we're going to build the individual models. As we're using SageMaker pre-built models, we need to provide the name of the
container then we can create it through the function `session.sagemaker_client.create_model`. 

```
import sagemaker
from sagemaker import get_execution_role
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.predictor import csv_serializer

# Create a SageMaker Session 
session = sagemaker.Session()

# Get the SageMaker execution role 
role = get_execution_role()

# Set a name for the linear model 
linear_model_name = "boston-update-linear-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Retrieve a container for the linear model 
linear_container = get_image_uri(session.boto_region_name, 'linear-learner')

# Set container for linear model 
linear_primary_container = {
    "Image": linear_container,
    "ModelDataUrl": linear.model_data
}

# Construct the linear model 
linear_model_info = session.sagemaker_client.create_model(
                                ModelName = linear_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = linear_primary_container)

# Set a name for the xgboost model 
xgb_model_name = "boston-update-xgboost-model" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Retrieve a container for the XGBoost model 
xgb_container = get_image_uri(session.boto_region_name, 'xgboost')

# Set container for the xgboost model 
xgb_primary_container = {
    "Image": xgb_container,
    "ModelDataUrl": xgb.model_data
}

# Construct the XGBoost model 
xgb_model_info = session.sagemaker_client.create_model(
                                ModelName = xgb_model_name,
                                ExecutionRoleArn = role,
                                PrimaryContainer = xgb_primary_container)
``` 

## Step 2. Deploying a combined model 

One can decide how much weight is going to be given to each of the models deployed by the value of the parameter `InitialVariantWeight`. For example, considering two models if one decides that the first one is going to have a weight of 80% and the second of 20%. Then, one assigns a weight of 4 and 1 to the model respectively, that 4/(4+1) is going to be the weight assigned to the first model. 

```
# Set a name for the endpoint 
combined_endpoint_config_name = "boston-combined-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Configure the endpoint 
combined_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = combined_endpoint_config_name,
                            ProductionVariants = [
                                { # First we include the linear model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": linear_model_name,
                                    "VariantName": "Linear-Model"
                                }, { # And next we include the xgb model
                                    "InstanceType": "ml.m4.xlarge",
                                    "InitialVariantWeight": 1,
                                    "InitialInstanceCount": 1,
                                    "ModelName": xgb_model_name,
                                    "VariantName": "XGB-Model"
                                }])

# Set a unique name for the combined model 
endpoint_name = "boston-combined-endpoint-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Create and deploy the endpoint 
endpoint_info = session.sagemaker_client.create_endpoint(
                    EndpointName = endpoint_name,
                    EndpointConfigName = combined_endpoint_config_name)

# Wait for endpoint to be deployed 
endpoint_dec = session.wait_for_endpoint(endpoint_name)
```

If at a given moment, we would like to know the properties of the combined model, we can just describe it `pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))`

## Step 3. Using the combined model 

We can briefly see how the models are behaving by comparing the prediction with the real value. 

```
for rec in range(10):

    # Call the endpoint 
    response = session.sagemaker_runtime_client.invoke_endpoint(
                                                EndpointName = endpoint_name,
                                                ContentType = 'text/csv',
                                                Body = ','.join(map(str, X_test.values[rec])))

    pprint(response)
    result = response['Body'].read().decode("utf-8")
    print(result)
    print(Y_test.values[rec])
```


## Step 4. Updating the endpoint 

After doing the A/B test to compare the models, suppose we decide to leave the Linear Model on production. We want to switch our endpoint from sending data to both the XGBoost model and the linear model to sending data only to the linear model. We can use SageMaker to update an endpoint to a new endpoint configuration, without shutting it down. Basically, SageMaker will set the endpoint to new characteristics. Once this new endpoint is running, SageMaker will switch the old endpoint so that it now points at the newly deployed model, making sure that this happens seamlessly in the background.

```
# Give a unique name to the linear model endpoint 
linear_endpoint_config_name = "boston-linear-endpoint-config-" + strftime("%Y-%m-%d-%H-%M-%S", gmtime())

# Set the endpoint configuration
linear_endpoint_config_info = session.sagemaker_client.create_endpoint_config(
                            EndpointConfigName = linear_endpoint_config_name,
                            ProductionVariants = [{
                                "InstanceType": "ml.m4.xlarge",
                                "InitialVariantWeight": 1,
                                "InitialInstanceCount": 1,
                                "ModelName": linear_model_name,
                                "VariantName": "Linear-Model"
                            }])

# Update the endpoint 
session.sagemaker_client.update_endpoint(EndpointName=endpoint_name, EndpointConfigName=linear_endpoint_config_name)

# Wait for the endpoint to be updated 
endpoint_dec = session.wait_for_endpoint(endpoint_name)

# Verify the new characteristics of the endpoint 
pprint(session.sagemaker_client.describe_endpoint(EndpointName=endpoint_name))
```

Finally, remember to always shut down the endpoint if it is no longer going to be in use `session.sagemaker_client.delete_endpoint(EndpointName = endpoint_name)`. 