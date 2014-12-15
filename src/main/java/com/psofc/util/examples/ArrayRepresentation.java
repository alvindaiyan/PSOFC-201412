package com.psofc.util.examples;

import com.psofc.*;
import com.psofc.singlepsofc.arrayrep.FeatureConstruction;
import com.psofc.singlepsofc.arrayrep.OperatorSelection;
import com.psofc.util.FCFunction;
import com.psofc.util.OutputYan;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;

import java.util.Random;

/**
 * Created by daiyan on 15/12/14.
 */
public class ArrayRepresentation extends BaseRepresentationRun
{

    String fname = "australian";
    String dir = "file_array";

    @Override
    public void start(long[] seeder)
    {
        ArrRecord arrRecord = new ArrRecord();

        /** start PSO */
        for(int r = 0; r < number_of_runs; r++)
        {
            long initTime = System.currentTimeMillis();

            System.out.println("*************************************** run of " + r + " *****************************************");

            RandomBing.Seeder.setSeed(seeder[r]);

            System.out.println("RandomBing.Seeder in  " + (r + 1) + " run is:  " + seeder[r]);

            Swarm s = new Swarm();
            s.setProblem(problem);
            // to set the classes and the attributes in feature selection();
            s.getProblem().setTraining(training);

            s.getProblem().setNumFolds(BaseRepresentationRun.TOTAL_FOLDS_NUMBER);
            s.getProblem().setFoldsTrain(foldsTrain);
            s.getProblem().setThreshold(0.6);
            s.setTopology(topology);
            s.setVelocityClamp(new VelocityClampBasic());

            for (int i = 0; i < number_of_particles; ++i)
            {
                Particle p = new Particle();
                p.setProblem(s.getProblem());
                p.setSize(dimension);
                p.setC1(c1);
                p.setC2(c2);
                s.addParticle(p);

            }
            // initial the swarm
            s.initialize();

            // start iterations
            for(int i = 0; i < number_of_iterations; i++)
            {
                s.iterate(w);
                double bestFitness = s.getParticle(0).getNeighborhoodFitness();

                System.out.println(bestFitness);

                int bestParticle = 0;
                for (int p = 0; p < number_of_particles; p++)
                {
                    if (s.getProblem().isBetter(
                            s.getParticle(p).getNeighborhoodFitness(),
                            bestFitness))
                    {
                        bestFitness = s.getParticle(p).getNeighborhoodFitness();
                        bestParticle = p;
                    }
                }

                arrRecord.gbestRunsIterations[r][i] = bestFitness;

                arrRecord.eroGbestRunsIterations[r][i] = bestFitness; // fitness = error;

				/**
                 * get the position of best results in the last generation in
				 * each run; to get the best solution for best fitness in each
				 * run
				 */
                if (i == number_of_iterations - 1)
                {
                    arrRecord.bestFitnessRuns[r] = arrRecord.gbestRunsIterations[r][i];
                    arrRecord.accTrainRunsYan[r] = 1.0 - arrRecord.eroGbestRunsIterations[r][i];

                    // System.arraycopy(s.getParticle(bestParticle).getNeighborhoodPosition().toArray(), 0, bestPositionRuns[r], 0, dimension);

                    for (int d = 0; d < s.getParticle(bestParticle)
                            .getNeighborhoodPosition().size(); d++)
                    {
                        arrRecord.bestPositionRuns[r][d] = s.getParticle(bestParticle)
                                .getNeighborhoodPosition(d);
                    }
                }

            } // end all iterations

            // summarise a iteration
            recordIteration(r, arrRecord, initTime);
        }
    }


    /**
     * calculate testing accuracy of the constructed feature
     * @param r
     * @param arrRecord
     * @param initTime
     */
    private void recordIteration(int r, ArrRecord arrRecord ,long initTime)
    {
        long estimatedTime = System.currentTimeMillis() - initTime;

        System.out.println("estimatedTime: " + estimatedTime);

        arrRecord.operatorsRuns[r] = OperatorSelection.getOpFromArray(arrRecord.bestPositionRuns[r]);

        double[] features = new double[noFeatures];
        int p = 0;

        System.out.println(arrRecord.bestPositionRuns[r].length);

        for (int i = 0; i < arrRecord.bestPositionRuns[r].length; i += 2)
        {
            features[p++] = arrRecord.bestPositionRuns[r][i];
        }

        Dataset temTrain = training.copy();
        temTrain = HelpDataset.removeFeatures(temTrain, features);
        temTrain = FCFunction.calConstructFeaBing(temTrain, arrRecord.operatorsRuns[r]);

        Dataset temTest = testing.copy();
        temTest = HelpDataset.removeFeatures(temTest, features);
        temTest = FCFunction.calConstructFeaBing(temTest, arrRecord.operatorsRuns[r]);

        for (int rm = 0; rm < temTest.size(); rm++)
        {
            Instance line = temTest.get(rm);
            arrRecord.constructedFeatureTtRuns[r][rm] = line.get(0);
        }

        for (int rm = 0; rm < temTrain.size(); rm++)
        {
            Instance line = temTrain.get(rm);
            arrRecord.constructedFeatureTrRuns[r][rm] = line.get(0);
        }


        for (char op : arrRecord.operatorsRuns[r])
        {
            System.out.print(op + " ");
        }
        System.out.println();

        MyClassifier mc = new MyClassifier(new Random(1));

        arrRecord.accTestRunsYan[r] = ((FeatureConstruction) problem).classify(
                temTrain, temTest);

        // decision tree test
        mc.ClassifierDT();
        try
        {
            arrRecord.accTrainRunsDT[r] = mc.classify(temTrain, temTrain);
        }
        catch (Error e)
        {
            arrRecord.accTrainRunsDT[r] = 0.0;
        }

        try
        {
            arrRecord.accTestRunsDT[r] = mc.classify(temTrain, temTest);
        }
        catch (Error e)
        {
            arrRecord.accTestRunsDT[r] = 0.0;
            arrRecord.DTNum1--;
        }

        // navie bayes test
        mc.ClassifierNB();
        try
        {
            arrRecord.accTrainRunsNB[r] = mc.classify(temTrain, temTrain);
        }
        catch (Error e)
        {
            arrRecord.accTrainRunsNB[r] = 0.0;
        }
        arrRecord.accTestRunsNB[r] = mc.classify(temTrain, temTest);

        // k-nearest neighbour test
        mc.ClassifierKNN();
        try
        {
            arrRecord.accTrainRunsKNN[r] = mc.classify(temTrain, temTrain);
        }
        catch (Exception e)
        {
            arrRecord.accTrainRunsKNN[r] = 0.0;
            e.printStackTrace();
        }

        try
        {
            arrRecord.accTestRunsKNN[r] = mc.classify(temTrain, temTest);
        }
        catch (Exception e)
        {
            arrRecord.accTestRunsKNN[r] = 0.0;
            arrRecord.KNNNum1--;
            e.printStackTrace();
        }



        // output to file
        OutputYan fcorg = new OutputYan(dir, fname);

        Dataset temp = fcorg.constructNewDataset(data, features, arrRecord.operatorsRuns[r]);
        try
        {
            arrRecord.CFOrgAccTrainingRunsDT[r] = fcorg.superDataTestingDT(temp)[0];
            System.out.println(arrRecord.CFOrgAccTrainingRunsDT[r]);
        }
        catch (Exception e)
        {
            arrRecord.CFOrgAccTrainingRunsDT[r] = 0.0;
            System.out.println("error DT");
        }

        try
        {
            arrRecord.CFOrgAccTestingRunsDT[r] = fcorg.superDataTestingDT(temp)[1];
        }
        catch (Exception e)
        {
            arrRecord.CFOrgAccTestingRunsDT[r] = 0.0;
            arrRecord.DTNum2--;
        }

        // try{
        arrRecord.CFOrgAccTrainingRunsKNN[r] = fcorg.superDataTestingKNN(temp)[0];
        // }catch(Exception e){CFOrgAccTrainingRunsKNN[r] = 0.0;
        // e.printStackTrace();}

        // try{
        arrRecord.CFOrgAccTestingRunsKNN[r] = fcorg.superDataTestingKNN(temp)[1];
        // }catch(Exception e){CFOrgAccTestingRunsKNN[r] = 0.0;
        // e.printStackTrace();}

        arrRecord.CFOrgAccTrainingRunsNB[r] = fcorg.superDataTestingNB(temp)[0];

        arrRecord.CFOrgAccTestingRunsNB[r] = fcorg.superDataTestingNB(temp)[1];

        arrRecord.CFOrgAccTrainingRunsYan[r] = fcorg.superDataTestingYan(temp)[0];

        arrRecord.CFOrgAccTestingRunsYan[r] = fcorg.superDataTestingYan(temp)[1];

        arrRecord.timeRuns[r] = estimatedTime;
        System.out.println("");
    }




    public class ArrRecord implements PrintResult.Record
    {
        double[][] gbestRunsIterations = new double[number_of_runs][number_of_iterations]; // get best fitness in each iterate in each run
        double[][] eroGbestRunsIterations = new double[number_of_runs][number_of_iterations];

        double[] bestFitnessRuns = new double[number_of_runs]; // get final bestfitnes in each run

        double[] accTestRunsYan = new double[number_of_runs]; // the best testing accuracy in each run
        double[] accTrainRunsYan = new double[number_of_runs];
        int YanNum1 = number_of_runs;
        int YanNum2 = number_of_runs;

        double[] accTestRunsDT = new double[number_of_runs]; // the best testing accuracy in each run
        double[] accTrainRunsDT = new double[number_of_runs];
        int DTNum1 = number_of_runs;
        int DTNum2 = number_of_runs;

        double[] accTestRunsKNN = new double[number_of_runs]; // the best testing accuracy in each run
        double[] accTrainRunsKNN = new double[number_of_runs];
        int KNNNum1 = number_of_runs;
        int KNNNum2 = number_of_runs;

        double[] accTestRunsNB = new double[number_of_runs]; // the best testing accuracy in each run
        double[] accTrainRunsNB = new double[number_of_runs];
        int NBNum1 = number_of_runs;
        int NBNum2 = number_of_runs;

        double[][] bestPositionRuns = new double[number_of_runs][dimension];// get the position of best results in each run;

        long[] timeRuns = new long[number_of_runs];

        double[] CFOrgAccTestingRunsDT = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsDT = new double[number_of_runs];

        double[] CFOrgAccTestingRunsKNN = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsKNN = new double[number_of_runs];

        double[] CFOrgAccTestingRunsNB = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsNB = new double[number_of_runs];

        double[] CFOrgAccTestingRunsYan = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsYan = new double[number_of_runs];

        char[][] operatorsRuns = new char[number_of_runs][dimension - 1];
        // temperory, this need to according the results of the selected operators

        double[][] constructedFeatureTrRuns = new double[number_of_runs][training_size];
        double[][] constructedFeatureTtRuns = new double[number_of_runs][testing_size];



    }

}
