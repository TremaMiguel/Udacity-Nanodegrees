# Machine Learning Engineer Nanodegree

The Machine Learning Engineer Nanodegree has as objective to deploy a Machine Learning Model using AWS stack, mostly Amazon SageMaker. The theory of how to do this can be found under the folder `Machine_Learning_in_Production`. Some practical projects are under the folder `Machine_Learning_Case_Studies`. 

## 1. Machine Learning in Production 

Five lessons are covered using Amazon SageMaker service: 

    1. Introduction to Deployment. Quick intro of basic deployment concepts, such as, what is an endpoint, what is a container, restful API. 

    2. Building Models in SageMaker. Configure a SageMaker estimator through the Low Level or the High Level Interface. Additionally, set up a Transform Job to execute data in batches.

    3. Deployment Using a Model. Configure a SageMaker endpoint through the Low Level or High Level Interface. Deploy a model with SageMaker Enpoint + AWS Lambda + AWS API Gateway. 

    4. Hyperparameter Tuning. Use the SageMaker Tuner for tuning Model hyperparameters throug the Low Level or High Level Interface. 

    5. Update a Model. Perform A/B testing to update a model already deployed (update the endpoint). Understand concepts like Blue-Green Deployment. 

## Machine Learning Case Studies 

Put in practice all the theory through different projects. 

    1. Population Segmentation. Dimensionality reduction through PCA and clustering with K-Means algorithm. 

![Segmentation_Example](Machine_Learning_Case_Studies/Population_Segmentation/figs/pca_2d_dim_reduction.png)

    2. Fraud Detection. Understand concepts like precision and recall. Tune SageMaker LinearLearner to achieve the highest recall or the highest precision or to deal with unbalance data. 

![Detection_Example](Machine_Learning_Case_Studies/Fraud_Detection/figs/precision_recall.png)

    3. Moon Data. Deploy a custom Pytorch Neural Network model to classify data. 



    4. Forecasting. Use SageMaker's DeepAR (Recurrent Neural Network) to forecast data. 

![Forecasting_Example](Machine_Learning_Case_Studies/Forecasting/figs/context_prediction_windows.png)

    5. Project. Built a custom Scikit Learn Model to detect plagiarism in text through Containtment and Longest Common Subsequence.  



