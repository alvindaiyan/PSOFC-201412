package com.psofc.util;

import com.psofc.MyClassifier;
import com.psofc.util.examples.BaseRepresentationRun;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

public class ClassificationPerformance
{
    private Dataset[] folds;
    private Dataset training;
    private Dataset testing;
    private MyClassifier classifier;

    public static final String FULL_SET_TRAINING_ACC_TAG = "fullSet.Trainacc()";
    public static final String FULL_SET_TESTING_ACC_TAG = "fullSet.Testacc()";


    /**
     *
     * @param data the data set to be tested
     * @param classifier the classifier used to
     */
    public ClassificationPerformance(Dataset data, MyClassifier classifier)
    {
        this.folds = data.folds((BaseRepresentationRun.TOTAL_FOLDS_NUMBER), new Random(BaseRepresentationRun.RANDOM_SEED));
        this.training = new DefaultDataset();
        this.testing = new DefaultDataset();
        this.classifier = classifier;
    }

    /**
     * To get classification performance of using all original features
     * @return Map
     */
    public Map<String, Double> getClassificationPerformance()
    {
        Map<String, Double> resultLookup = new HashMap<String, Double>();


        int[] tr = {0, 2, 3, 5, 6, 8, 9};
        int[] te = {1, 4, 7}; // 7, 4 and 6,5 changes
        for (int i = 0; i < tr.length; i++)
        {
            training.addAll(folds[tr[i]]);
        }
        for (int i = 0; i < te.length; i++)
        {
            testing.addAll(folds[te[i]]);
        }

        double fullTrain = this.classifier.fullclassify(training, training);
        double fullTest = this.classifier.fullclassify(training, testing);

        resultLookup.put(FULL_SET_TRAINING_ACC_TAG, fullTrain);
        resultLookup.put(FULL_SET_TESTING_ACC_TAG, fullTest);

        return resultLookup;
    }


    public Dataset[] getFolds()
    {
        return folds;
    }

    public Dataset getTraining()
    {
        return training;
    }

    public Dataset getTesting()
    {
        return testing;
    }

    public MyClassifier getClassifier()
    {
        return classifier;
    }
}