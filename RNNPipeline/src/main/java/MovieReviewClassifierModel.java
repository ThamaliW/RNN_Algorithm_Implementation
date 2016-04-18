/**
 * Created by thamali on 2/8/16.
 */
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.ml.PredictionModel;
import org.apache.spark.ml.param.ParamMap;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import java.io.*;


//Model produced by Prediction mddel
public class MovieReviewClassifierModel extends PredictionModel<Object, MovieReviewClassifierModel> implements  Serializable{

    private static final long serialVersionUID = 1L;
    private MultiLayerNetwork net;
    private int maxLength;
    int vectorSize=300;

    MovieReviewClassifierModel (MultiLayerNetwork net) throws Exception {
        this.net=net;
        maxLength=DataIterator2.maximumLength;
    }

    @Override
    public MovieReviewClassifierModel copy(ParamMap args0) {
        return null;
    }

    @Override
    public String uid() {
        return "MovieReviewClassifierModel";
    }

    @Override
    public double predict(Object o) {//predict row by row

        int prediction=0;//Variable to store prediction
        DenseVector vector=(DenseVector)o;//Convert to a vector

        double[] featureArray=vector.toArray();//Get features to an array
        int lengthOfFeatureArray=featureArray.length;
        INDArray featureINDArray=Nd4j.create(vectorSize,lengthOfFeatureArray);//Create an INDArray to store features

        for( int j=0; j<featureArray.length; j++ ){//Loop to add feature array elements to the INDArray
            Double token=featureArray[j];
            featureINDArray.put(new INDArrayIndex[]{ NDArrayIndex.point(j)}, token);
        }

        INDArray featureMaskArray=Nd4j.zeros(maxLength);//Create a Mask array for masking features

        for(int i=0;i<lengthOfFeatureArray;i++){
            featureMaskArray.putScalar(i,1.0);
        }

        INDArray labelsMaskINDArray = Nd4j.zeros(maxLength);//Create a mask array for label Masks
        int lastIdx = Math.min(lengthOfFeatureArray,maxLength);
        labelsMaskINDArray.putScalar(lengthOfFeatureArray-1,1.0);

        INDArray array= net.output(featureINDArray,false,featureMaskArray,labelsMaskINDArray);//Predict the label using INDArray.output method


        double zeroprob=array.getDouble(2*maxLength-1);
        double oneprob=array.getDouble(2*maxLength-2);
        if(zeroprob>oneprob){prediction=0;}
        else{prediction=1;}
        System.out.println(prediction);
        System.out.println(zeroprob);
        System.out.println(oneprob);
        return prediction;
    }
}