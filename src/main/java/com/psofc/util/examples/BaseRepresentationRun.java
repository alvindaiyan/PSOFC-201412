package com.psofc.util.examples;

import com.psofc.MyClassifier;
import com.psofc.Problem;
import com.psofc.Topology;
import com.psofc.TopologyRing;
import com.psofc.singlepsofc.arrayrep.FeatureConstruction;
import com.psofc.util.ClassificationPerformance;
import com.psofc.util.ReadResults;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.tools.data.FileHandler;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Map;
import java.util.Random;

/**
 * Created by daiyan on 15/12/14.
 */
public abstract class BaseRepresentationRun
{
    public static final int TOTAL_FOLDS_NUMBER = 10;
    public static final long RANDOM_SEED = 100;




    protected int dimension = 0;
    protected int number_of_runs = 1;
    protected double w = 0.729844;
    protected double c1 = 1.49618, c2 = 1.49618;
    protected int number_of_particles = 30;
    protected int number_of_iterations = 50;
    protected Topology topology = new TopologyRing(30);
    protected Dataset data;
    protected Problem problem;

    protected Dataset training;
    protected Dataset testing;
    protected int training_size;
    protected int testing_size;


    protected double fullTrain;
    protected double fullTest;

    protected PrintResult.Record record;

    protected Dataset[] foldsTrain;
    protected int noFeatures;
    protected DecimalFormat df = new DecimalFormat("##.##");
    protected DecimalFormat dg = new DecimalFormat("##.#E0");


    public boolean setup(String fname, String dir)
    {
        try
        {
            // loading the files from the data folder
            this.noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname + "/noFeatures.txt"));
            this.data = FileHandler.loadDataset(new File("Data/" + fname + "/Data.data"), noFeatures, ",");

            this.dimension = noFeatures * 2 - 1;

            this.problem = new FeatureConstruction();
            // System.out.println("Problem = new FeatureConstruction()");

            this.problem.setMyclassifier(new MyClassifier(new Random(1)));
            this.problem.getMyclassifier().ClassifierDT(); // new ClassifierKNN() can be used


            // get the accuracy performance by using the original data
            ClassificationPerformance originalAcc = new ClassificationPerformance(data, problem.getMyclassifier());

            Map<String, Double> p = originalAcc.getClassificationPerformance();

            for(Map.Entry<String, Double> entry : p.entrySet())
            {
                System.out.println(entry.getKey() + ": " + entry.getValue());
            }

            this.foldsTrain = originalAcc.getTraining().folds(TOTAL_FOLDS_NUMBER, new Random(1));

            this.fullTrain = p.get(ClassificationPerformance.FULL_SET_TRAINING_ACC_TAG);
            this.fullTest = p.get(ClassificationPerformance.FULL_SET_TESTING_ACC_TAG);

            this.training = originalAcc.getTraining();
            this.testing = originalAcc.getTesting();

            this.training_size = originalAcc.getTraining().size();
            this.testing_size = originalAcc.getTesting().size();

        }
        catch (IOException e)
        {
            e.printStackTrace();
            return false;
        }

        return true;
    }


    public long[] getRandomSeeder()
    {
        long[] Seeder = new long[number_of_runs];
        for (int r = 0; r < number_of_runs; r++)
        {
            Seeder[r] = r * r * r * 135 + r * r * 246 + 78;
        }
        return Seeder;
    }





    public abstract void start(long[] seeder);
}
