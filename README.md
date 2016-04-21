# RNN_Algorithm_Implementation
We have worked on a project to add Recurrent Neural Network(RNN) algorithm to machine learner. RNN is one of deep learning algorithms with record breaking accuracy. For more information on RNN please refer link[1].

We have decided to use deeplearning4j which is an open source deep learning library scalable on spark and Hadoop.

Since there is a plan to add spark pipeline to machine Learner, we have decided to use spark pipeline concept to our project.

I have designed an architecture for the RNN implementation. 

This architecture is developed to be compatible with spark pipeline.

Data set is taken in csv format and then it is converted to spark data frame since apache spark works mostly with data frames.

Next step is a transformer which is needed to tokenize the sequential data. A tokenizer is basically used for take a sequence of data and break it into individual units. For example, it can be used to break the words in a sentence to words.

Next step is again a transformer used to converts tokens to vectors. This must be done because the features should be added to spark pipeline in org.apache.spark.mllib.linlag.VectorUDT format.

Next, the transformed data are fed to the data set iterator. This is an object of a class which implement org.deeplearning4j.datasets.iterator.DataSetIterator. The dataset iterator traverses through a data set and prepares data for neural networks.

Next component is the RNN algorithm model which is an estimator. The iterated data from data set iterator is fed to RNN and a model is generated. Then this model can be used for predictions.

We  decided to complete this project in two steps :

    First create a spark pipeline program containing the steps in machine learner(uploading dataset, generate model, calculating accuracy and prediction) and check whether the project is feasible.

    Next add the algorithm to ML

we have created a spark program to prove the feasibility of adding the RNN algorithm to machine learner.
This program demonstrates all the steps in machine learner:

Uploading a dataset

Selecting the hyper parameters for the model

Creating a RNN model using data and training the model

Calculating the accuracy of the model

Saving the model(As a serialization object)

predicting using the model

This program is based on deeplearning4j and apache spark pipeline. Deeplearning4j was used as the deep learning library for recurrent neural network algorithm. As the program should be based on the Spark pipeline, the main challenge was to use deeplearning4j library with spark pipeline. The components used in the spark pipeline should be compatible with spark pipeline. For other components which are not compatible with spark pipeline, we have to wrap them with a org.apache.spark.predictionModel object.

We have designed a pipeline with sequence of stages (transformers and estimators):

1. Tokenizer:Transformer-Split each sequential data to tokens.(For example, in sentiment analysis, split text into words)

2. Vectorizer :Transformer-Transforms features into vectors.

3. RNN algorithm :Estimator -RNN algorithm which trains on a data frame and produces a RNN model

4. RNN model : Transformer- Transforms data frame with features to data frame with predictions. 


Number of epochs-10

Number of iterations- 1

Learning rate-0.02

We used the aclImdb sentiment analysis data set for this program and with the above hyper parameters, we could achieve 60% accuracy. And we are trying to improve the accuracy and efficiency of our algorithm.


