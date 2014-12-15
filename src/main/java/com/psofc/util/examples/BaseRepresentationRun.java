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
import java.util.Map;
import java.util.Random;

/**
 * Created by daiyan on 15/12/14.
 */
public abstract class BaseRepresentationRun
{
    public static final int TOTAL_FOLDS_NUMBER = 10;
    public static final long RANDOM_SEED = 100;




    private int dimension = 0;
    private int number_of_runs = 50;
    private double w = 0.729844;
    private double c1 = 1.49618, c2 = 1.49618;
    private int number_of_particles = 30;
    private int number_of_iterations = 100;
    private Topology topology = new TopologyRing(30);
    private Dataset data;
    Problem problem;

    public void setup()
    {
        try
        {
            // loading the files from the data folder
            String fname = "australian";
            String dir = "file_array";

            int noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname + "/noFeatures.txt"));
            this.data = FileHandler.loadDataset(new File("Data/" + fname + "/Data.data"), noFeatures, ",");

            this.dimension = noFeatures * 2 - 1;

            this.problem = new FeatureConstruction();
            // System.out.println("Problem = new FeatureConstruction()");

            this.problem.setMyclassifier(new MyClassifier(new Random(1)));
            this.problem.getMyclassifier().ClassifierDT(); // new ClassifierKNN() can be used


            // get the accuracy performance by using the original data
            ClassificationPerformance originalAcc = new ClassificationPerformance(data, problem.getMyclassifier());

            for(Map.Entry<String, Double> entry : originalAcc.getClassificationPerformance().entrySet())
            {
                System.out.println(entry.getKey() + ": " + entry.getValue());
            }

            Dataset[] foldsTrain = originalAcc.getTraining().folds(TOTAL_FOLDS_NUMBER, new Random(1));






        }
        catch (IOException e)
        {
            e.printStackTrace();
        }
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



    public abstract void start();
}
