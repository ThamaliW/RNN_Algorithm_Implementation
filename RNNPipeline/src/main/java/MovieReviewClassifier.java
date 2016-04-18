/**
 * Created by thamali on 2/8/16.
 */


import org.apache.spark.ml.Predictor;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.sql.DataFrame;
import org.deeplearning4j.datasets.iterator.AsyncDataSetIterator;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.eval.Evaluation;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.GradientNormalization;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.Updater;
import org.deeplearning4j.nn.conf.layers.GravesLSTM;
import org.deeplearning4j.nn.conf.layers.RnnOutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import java.io.*;


//This class extends predictor which is a Single-label binary or multiclass classifier which can output class conditional probabilities.
public class MovieReviewClassifier extends Predictor<Object, MovieReviewClassifier, MovieReviewClassifierModel> implements Serializable{

    private static final long serialVersionUID = 1L;


    @Override
    public MovieReviewClassifier copy(ParamMap arg0) {
        return null;
    }


    @Override
    //An immutable unique ID for the object and its derivatives
    public String uid() {
        return "MovieReviewClassifier";
    }

    @Override
    public  MovieReviewClassifierModel train(DataFrame dataFrame)
    {
    dataFrame.show();

        int batchSize = 50;     //Number of examples in each minibatch
        int vectorSize = 300;   //Size of the word vectors. 300 in the Google News model
        int nEpochs =10;        //Number of epochs (full passes of training data) to train on

        //Set up network configuration
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT).iterations(1)
                .updater(Updater.RMSPROP).seed(12345)
                .regularization(true).l2(1e-5)
                .weightInit(WeightInit.XAVIER)
                .gradientNormalization(GradientNormalization.ClipElementWiseAbsoluteValue).gradientNormalizationThreshold(1.0)
                .learningRate(0.018)
                .list(2)
                .layer(0, new GravesLSTM.Builder().nIn(vectorSize).nOut(200)
                        .activation("softsign").build())
                .layer(1, new RnnOutputLayer.Builder().activation("softmax")
                        .lossFunction(LossFunctions.LossFunction.MCXENT).nIn(200).nOut(2).build())
                .pretrain(false).backprop(true).build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();
        net.setListeners(new ScoreIterationListener(1));

        //DataSetIterators
        //Using AsyncDataSetIterator to do data loading in a separate thread; this may improve performance vs. waiting for data to load

        DataSetIterator train = new AsyncDataSetIterator(new DataIterator2(dataFrame,batchSize),1);


        System.out.println("Starting training");
        for( int i=0; i<nEpochs; i++ ){
            net.fit(train);
            train.reset();
            System.out.println("Epoch " + i + " complete");
       }


        //Saving the model
        FileOutputStream fos = null;
        try {
            fos = new FileOutputStream("/home/thamali/RNN3/WordVec/M/output1");
            ObjectOutputStream oos = new ObjectOutputStream(fos);
            oos.writeObject(net);

            oos.close();
        } catch (Exception e) {
            e.printStackTrace();
        }


        //Wrap the model for pipeline usage
        MovieReviewClassifierModel svmModelWrapper = null;
        try {
            svmModelWrapper = new MovieReviewClassifierModel(net);
        } catch (Exception e) {
            e.printStackTrace();
        }

        return svmModelWrapper;

    }





}


