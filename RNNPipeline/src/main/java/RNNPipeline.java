/**
 * Created by thamali on 2/9/16.
 */
import org.apache.commons.lang3.ArrayUtils;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.*;
import org.apache.spark.mllib.evaluation.MulticlassMetrics;
import org.apache.spark.mllib.linalg.Matrix;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.SQLContext;


public class RNNPipeline {
    final static String RESPONSE_VARIABLE =  "s";
    final static String INDEXED_RESPONSE_VARIABLE =  "indexedClass";
    final static String PREDICTION = "prediction";
    final static String PREDICTION_LABEL = "predictionLabel";


    public static void main(String[] args) {

        SparkConf sparkConf = new SparkConf();
        sparkConf.setAppName("test-client").setMaster("local[2]");
        sparkConf.set("spark.driver.allowMultipleContexts", "true");
        JavaSparkContext javaSparkContext = new JavaSparkContext(sparkConf);
        SQLContext sqlContext = new SQLContext(javaSparkContext);

        // Convert data in csv format to Spark data frame
        DataFrame dataFrame = sqlContext.read().format("com.databricks.spark.csv")
                .option("inferSchema", "true")
                .option("header", "true")
                .load("/trndt3.csv");

        // Split in to train/test data
        double [] dataSplitWeights = {0.7,0.3};
        DataFrame[] data = dataFrame.randomSplit(dataSplitWeights);

        // Get predictor variable names
        String [] predictors = dataFrame.columns();
        predictors = ArrayUtils.removeElement(predictors, RESPONSE_VARIABLE);
        String featureColumnName=predictors[0];


        //Create a tokenizer object to transform data
        System.out.println("Tokenizing Started");
        Tokenizer tokenizer = new Tokenizer().setInputCol(featureColumnName).setOutputCol("text");
        System.out.println("Tokenizing Completed");

        //Create a word2vec model which is the tokenizer
        System.out.println("Vectorizing Started");
        Word2Vec word2Vec = new Word2Vec()
                .setInputCol("text")
                .setOutputCol("result")
                .setVectorSize(300)
                .setMinCount(5);
        System.out.println("Vectorizing Completed");

        // Encode labels
        StringIndexerModel labelIndexer = new StringIndexer().setInputCol(RESPONSE_VARIABLE)
                .setOutputCol(INDEXED_RESPONSE_VARIABLE)
                .fit(data[0]);


        // Convert indexed labels back to original labels (decode labels).
        IndexToString labelConverter = new IndexToString().setInputCol(PREDICTION)
                .setOutputCol(PREDICTION_LABEL)
                .setLabels(labelIndexer.labels());


        // ======================== Train ========================



        MovieReviewClassifier mrClassifier = new MovieReviewClassifier().setLabelCol(INDEXED_RESPONSE_VARIABLE).setFeaturesCol("result");



        // Fit the pipeline for training.
        Pipeline pipeline = new Pipeline().setStages(new PipelineStage[] { labelIndexer, tokenizer,word2Vec,mrClassifier, labelConverter});
        PipelineModel pipelineModel = pipeline.fit(data[0]);


        // ======================== Validate ========================
        DataFrame predictions = pipelineModel.transform(data[1]);
        predictions.show();

        // Confusion Matrix
        MulticlassMetrics metrics = new MulticlassMetrics(predictions.select(PREDICTION, INDEXED_RESPONSE_VARIABLE));
        Matrix confusionMatrix = metrics.confusionMatrix();

        // Accuracy Measures
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator().setLabelCol(INDEXED_RESPONSE_VARIABLE)
                .setPredictionCol(PREDICTION)
                .setMetricName("precision");
        double accuracy = evaluator.evaluate(predictions);

        System.out.println("===== Confusion Matrix ===== \n" + confusionMatrix + "\n============================");
        System.out.println("Accuracy = " + accuracy*100+"%");


    }


}