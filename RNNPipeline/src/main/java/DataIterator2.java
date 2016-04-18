import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.sql.DataFrame;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.execution.datasources.Partition;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.ArrayList;
import java.util.List;
import java.util.NoSuchElementException;

/**
 * Created by thamali on 2/23/16.
 */
public class DataIterator2  implements DataSetIterator {
    static  int maximumLength;
    private int vectorSize=300;
    private int truncateLength=600;
    private int cursor = 0;
    private int length;
    private DataFrame dataframe;
    private int batchSize;


    public DataIterator2(DataFrame dataFrame,int batchSize){
        this.dataframe=dataFrame;
        this.batchSize=batchSize;
        JavaRDD<Row> r=dataframe.toJavaRDD();
        this.length=(int)r.count();//calculate the length of data frame
        dataframe.show();
    }

    @Override
    public DataSet next(int num) {//Returns the next data set when the batch size is given
        if (cursor >= length) throw new NoSuchElementException();
        return nextDataSet(num);
    }


    private DataSet nextDataSet(int num) {
        List<double[]> reviews = new ArrayList(num);
        double[] labelarray = new double[num];

        DataFrame reviewFrame=dataframe.select("result");//Get the vectorized feature column
        JavaRDD<Row> reviewRow =reviewFrame.toJavaRDD();
        List<Row> rows= reviewRow.collect();


        DataFrame labelFrame=dataframe.select("indexedClass");//get the indexed response variable column
        JavaRDD<Row> labelRow=labelFrame.toJavaRDD();
        List<Row> rows2=labelRow.collect();


        int maxLength=0;
        for(int i=0;i<num && cursor<totalExamples();i++){
            DenseVector vector=(DenseVector) rows.get(cursor).get(0);
            double [] arr=vector.toArray();
            int arrayLength=arr.length;
            maxLength = Math.max(maxLength,arrayLength);
            reviews.add(arr);
            labelarray[i]=rows2.get(cursor).getDouble(0);
            cursor++;
        }

        if(maxLength > truncateLength) maxLength = truncateLength;

        INDArray features = Nd4j.create(reviews.size(), vectorSize, maxLength);
        INDArray labels = Nd4j.create(reviews.size(), 2, maxLength);    //Two labels: positive or negative
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(reviews.size(), maxLength);
        INDArray labelsMask = Nd4j.zeros(reviews.size(), maxLength);
        maximumLength=maxLength;

        int[] temp = new int[2];
        for( int i=0; i<reviews.size(); i++ ){
            double[] tokens = reviews.get(i);
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.length && j<maxLength; j++ ){

                Double token=tokens[j];

                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all(), NDArrayIndex.point(j)}, token);
                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }

            Double d=labelarray[i];
            int idx =d.intValue();
            int lastIdx = Math.min(tokens.length,maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);

    }

    @Override
    public int totalExamples() {
        return length;
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        this.cursor=0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return this.length;
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor dataSetPreProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return null;
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }
}
