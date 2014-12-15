/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc.util.examples;

import java.io.*;
import java.text.DecimalFormat;
import java.text.NumberFormat;
import java.util.Random;

import com.psofc.HelpDataset;
import com.psofc.MyClassifier;
import com.psofc.Particle;
import com.psofc.Problem;
import com.psofc.RandomBing;
import com.psofc.Swarm;
import com.psofc.Topology;
import com.psofc.TopologyRing;
import com.psofc.VelocityClampBasic;
import com.psofc.singlepsofc.pair.FeatureConstruction;
import com.psofc.singlepsofc.pair.OperatorSelection;
import com.psofc.util.FCFunction;
import com.psofc.util.LatexFormat;
import com.psofc.util.NewMath;
import com.psofc.util.OrgClassification;
import com.psofc.util.OutputYan;
import com.psofc.util.ReadResults;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.tools.data.FileHandler;

@Deprecated
public class Main_Pair
{

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) throws Exception
    {

        String fname = "australian";
        String dir = "file_pair";

        int noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname
                + "/noFeatures.txt"));
        Dataset data = FileHandler.loadDataset(new File("Data/" + fname
                + "/Data.data"), noFeatures, ",");
        int dimension = noFeatures;
        int numFolds = 10;

        int number_of_runs = 50;
        double w = 0.729844;
        double c1 = 1.49618, c2 = 1.49618;
        int number_of_particles = 30;
        int number_of_iterations = 100;
        Topology topology = new TopologyRing(30);

        Problem problem = new FeatureConstruction(); // System.out.println("Problem = new FeatureConstruction()");

        problem.setMyclassifier(new MyClassifier(new Random(1))); // new
        // ClassifierKNN()
        problem.getMyclassifier().ClassifierDT();

        /**
         * To get classification performance of using all original features
         */
        Dataset[] folds = data.folds((10), new Random(100));
        Dataset training = new DefaultDataset();
        Dataset testing = new DefaultDataset();
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

        double fullTrain = problem.getMyclassifier().fullclassify(training,
                training);
        double fullTest = problem.getMyclassifier().fullclassify(training,
                testing);
        System.out.println("fullSet.Trainacc() " + fullTrain);
        System.out.println("fullSet.Testacc()   " + fullTest);

        Dataset[] foldsTrain = training.folds(numFolds, new Random(1));

        /** Sava the results */
        double[][] gbestRunsIterations = new double[number_of_runs][number_of_iterations];
        // get best fitness in each iterate in each run
        double[][] eroGbestRunsIterations = new double[number_of_runs][number_of_iterations];

        double[] bestFitnessRuns = new double[number_of_runs]; // get final
        // bestfitnes in each run

        double[] accTestRunsYan = new double[number_of_runs];
        // the best testing accuracy in each run
        double[] accTrainRunsYan = new double[number_of_runs];
        int YanNum1 = number_of_runs;
        int YanNum2 = number_of_runs;

        double[] accTestRunsDT = new double[number_of_runs]; // the best testing
        // accuracy in each run
        double[] accTrainRunsDT = new double[number_of_runs];
        int DTNum1 = number_of_runs;
        int DTNum2 = number_of_runs;

        double[] accTestRunsKNN = new double[number_of_runs];
        // the best testing accuracy in each run
        double[] accTrainRunsKNN = new double[number_of_runs];
        int KNNNum1 = number_of_runs;
        int KNNNum2 = number_of_runs;

        double[] accTestRunsNB = new double[number_of_runs];
        // the best testing accuracy in each run
        double[] accTrainRunsNB = new double[number_of_runs];
        int NBNum1 = number_of_runs;
        int NBNum2 = number_of_runs;

        // get the position of best results in each run;
        double[][] bestPositionRuns = new double[number_of_runs][dimension];


        long[] timeRuns = new long[number_of_runs];

        double[] CFOrgAccTestingRunsDT = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsDT = new double[number_of_runs];

        double[] CFOrgAccTestingRunsKNN = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsKNN = new double[number_of_runs];

        double[] CFOrgAccTestingRunsNB = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsNB = new double[number_of_runs];

        double[] CFOrgAccTestingRunsYan = new double[number_of_runs];
        double[] CFOrgAccTrainingRunsYan = new double[number_of_runs];

        char[][] operatorsRuns = new char[number_of_runs][noFeatures - 1]; // temperory, this need to
        // according the results
        // of the selected
        // operators

        /** Produce the random seeder */
        long[] Seeder = new long[number_of_runs];
        for (int r = 0; r < number_of_runs; r++)
        {
            Seeder[r] = r * r * r * 135 + r * r * 246 + 78;
        }

        /** start PSO */
        for (int r = 0; r < number_of_runs; ++r)
        {
            long initTime = System.currentTimeMillis();

            System.out
                    .println("**********************************************************************************");

            RandomBing.Seeder.setSeed(Seeder[r]);

            System.out.println("RandomBing.Seeder in  " + (r + 1)
                    + " run is:  " + Seeder[r]);

            Swarm s = new Swarm();
            s.setProblem(problem);
            s.getProblem().setTraining(training); // to set the classes and the
            // attributs in feature
            // selection();
            s.getProblem().setNumFolds(numFolds);
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
            s.initialize();

            for (int i = 0; i < number_of_iterations; ++i)
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

                gbestRunsIterations[r][i] = bestFitness;

                eroGbestRunsIterations[r][i] = bestFitness; // // fitness =
                // error;

				/*
                 * / get the position of best results in the last generation in
				 * each run; to get the best solution for best fitness in each
				 * run
				 */
                if (i == number_of_iterations - 1)
                {
                    bestFitnessRuns[r] = gbestRunsIterations[r][i];
                    accTrainRunsYan[r] = 1.0 - eroGbestRunsIterations[r][i];
                    // System.arraycopy(s.getParticle(bestParticle).getNeighborhoodPosition().toArray(),
                    // 0, bestPositionRuns[r], 0, dimension);
                    for (int d = 0; d < s.getParticle(bestParticle)
                            .getNeighborhoodPosition().size(); d++)
                    {
                        bestPositionRuns[r][d] = s.getParticle(bestParticle)
                                .getNeighborhoodPosition(d);
                    }
                }

            } // end all iterations

			/*
			 * calculate testing accuracy of the constructed feature
			 */

            long estimatedTime = System.currentTimeMillis() - initTime;
            System.out.println("estimatedTime: " + estimatedTime);

            operatorsRuns[r] = OperatorSelection.getOp(bestPositionRuns[r]);

            Dataset temTrain = training.copy();
            temTrain = HelpDataset
                    .removeFeatures(temTrain, bestPositionRuns[r]);
            temTrain = FCFunction.calConstructFeaBing(temTrain, operatorsRuns[r]);

            Dataset temTest = testing.copy();
            temTest = HelpDataset.removeFeatures(temTest, bestPositionRuns[r]);
            temTest = FCFunction.calConstructFeaBing(temTest, operatorsRuns[r]);

            // for(char opp:operators)System.out.print(opp + " " );
            // System.out.println();
            //

            // for(int rm = 0; rm < temTest.size(); rm++){
            // Instance line = temTest.get(rm);
            // // for(int i = 0; i < 15; i++){
            // // System.out.print(line.get(i) + ",");
            // // }
            // System.out.println("value: " + line.get(0));
            // }

            MyClassifier mc = new MyClassifier(new Random(1));

            accTestRunsYan[r] = ((FeatureConstruction) problem).classify(
                    temTrain, temTest);
            mc.ClassifierDT();

            try
            {
                accTrainRunsDT[r] = mc.classify(temTrain, temTrain);
            }
            catch (Error e)
            {
                accTrainRunsDT[r] = 0.0;
            }

            try
            {
                accTestRunsDT[r] = mc.classify(temTrain, temTest);
            }
            catch (Exception e)
            {
                DTNum1--;
                accTestRunsDT[r] = 0;
            }
            catch (Error e)
            {
                DTNum1--;
                accTestRunsDT[r] = 0;
            }

            mc.ClassifierNB();
            try
            {
                accTrainRunsNB[r] = mc.classify(temTrain, temTrain);
            }
            catch (Error e)
            {
                accTrainRunsNB[r] = 0.0;
            }

            try
            {
                accTestRunsNB[r] = mc.classify(temTrain, temTest);
            }
            catch (Exception e)
            {
                System.out.println("null");
                NBNum1--;
                accTestRunsNB[r] = 0;
            }

            mc.ClassifierKNN();
            try
            {
                accTrainRunsKNN[r] = mc.classify(temTrain, temTrain);
            }
            catch (Exception e)
            {
                e.printStackTrace();
                accTrainRunsKNN[r] = 0.0;
            }

            try
            {
                accTestRunsKNN[r] = mc.classify(temTrain, temTest);
            }
            catch (Exception e)
            {
                System.out.println("null");
                KNNNum1--;
                accTestRunsKNN[r] = 0;
            }

            OutputYan fcorg = new OutputYan(dir, fname);

            Dataset temp = fcorg.constructNewDataset(data, bestPositionRuns[r],
                    operatorsRuns[r]);

            try
            {
                CFOrgAccTrainingRunsDT[r] = fcorg.superDataTestingDT(temp)[0];
                CFOrgAccTestingRunsDT[r] = fcorg.superDataTestingDT(temp)[1];
            }
            catch (Exception e)
            {
                System.out.println("nullDT");
                CFOrgAccTestingRunsDT[r] = 0;
                DTNum2--;
            }
            catch (Error e)
            {
                System.out.println("errorDT");
                CFOrgAccTestingRunsDT[r] = 0;
                DTNum2--;
            }

            try
            {
                CFOrgAccTrainingRunsKNN[r] = fcorg.superDataTestingKNN(temp)[0];
                CFOrgAccTestingRunsKNN[r] = fcorg.superDataTestingKNN(temp)[1];
            }
            catch (Exception e)
            {
                System.out.println("null");
                CFOrgAccTestingRunsKNN[r] = 0;
                KNNNum2--;
            }

            try
            {
                CFOrgAccTrainingRunsNB[r] = fcorg.superDataTestingNB(temp)[0];

                CFOrgAccTestingRunsNB[r] = fcorg.superDataTestingNB(temp)[1];
            }
            catch (Exception e)
            {
                System.out.println("null");
                CFOrgAccTestingRunsNB[r] = 0;
                NBNum2--;
            }

            try
            {
                CFOrgAccTrainingRunsYan[r] = fcorg.superDataTestingYan(temp)[0];
                CFOrgAccTestingRunsYan[r] = fcorg.superDataTestingYan(temp)[1];
            }
            catch (Exception e)
            {
                System.out.println("null");
                CFOrgAccTestingRunsYan[r] = 0;
                YanNum2--;
            }

            timeRuns[r] = estimatedTime;
            System.out.println("");
        }// end all runsn

		/* All finished here */
        DecimalFormat df = new DecimalFormat("##.##");
        DecimalFormat dg = new DecimalFormat("##.#E0");

		/*
		 * // get best testing accuracy and its feature size If two jobs produce
		 * the same accuracies, the one with smaller size will be regarded as
		 * the better one ;
		 */
        int bestRun = 0;
        double bestAccTest = 0.0;
        for (int r = 0; r < number_of_runs; r++)
        {
            if (accTestRunsYan[r] > bestAccTest)
            {
                bestAccTest = accTestRunsYan[r];
                bestRun = r;
            }
        }

        System.out
                .println("Fitness                 TrainAcc            Acc_Test  (in each run)");
        for (int r = 0; r < number_of_runs; ++r)
        {
            // System.out.println("The" + (j + 1) + "Run");
            System.out.println(bestFitnessRuns[r] + "   " + accTrainRunsYan[r]
                    + "          " + accTestRunsYan[r]);
        }

        System.out.println("");
        System.out.println("Job for the Best testing acc is:  " + (bestRun + 1)
                + " Run (index+1) ");
        System.out.println("position for the Best testing accuracy is: ");
        int o = 0;
        char[] bestOps = OperatorSelection.getOp(bestPositionRuns[bestRun]);
        String op = "";
        String features = "";
        for (int d = 0; d < dimension; d++)
        {
            if (bestPositionRuns[bestRun][d] >= 0.5)
            {
                String operator = bestOps[o++] + ", ";
                System.out.print(bestPositionRuns[bestRun][d] + " ( f" + d
                        + " ) " + operator + " ");
                op += operator;
                features += " ( f" + d + " ) " + " ";
            }
        }

        System.out.println();
        System.out.println(bestPositionRuns[bestRun][7]);
        System.out.println("feature used: " + (o));
        System.out.println("feature used2: " + (features));
        System.out.println("operators used: " + op);
        System.out.println("");
        System.out.println("Best testing accuracy: " + bestAccTest);

        System.out.println("");
        double aveAccTestYan = NewMath.Mean_STD(accTestRunsYan, YanNum1)[0];
        double stdAccTestYan = NewMath.Mean_STD(accTestRunsYan, YanNum1)[1];
        System.out.println("Yan Average testing accuracy: " + aveAccTestYan);
        System.out.println("Yan Standard Deviation of testing accuracy: "
                + stdAccTestYan);

        System.out.println("");
        double aveAccTestDT = NewMath.Mean_STD(accTestRunsDT, DTNum1)[0];
        double stdAccTestDT = NewMath.Mean_STD(accTestRunsDT, DTNum1)[1];
        System.out.println("DT Average testing accuracy: " + aveAccTestDT);
        System.out.println("DT Standard Deviation of testing accuracy: "
                + stdAccTestDT);

        System.out.println("");
        double aveAccTestNB = NewMath.Mean_STD(accTestRunsNB, NBNum1)[0];
        double stdAccTestNB = NewMath.Mean_STD(accTestRunsNB, NBNum1)[1];
        System.out.println("NB Average testing accuracy: " + aveAccTestNB);
        System.out.println("NB Standard Deviation of testing accuracy: "
                + stdAccTestNB);

        System.out.println("");
        double aveAccTestKNN = NewMath.Mean_STD(accTestRunsKNN, KNNNum1)[0];
        double stdAccTestKNN = NewMath.Mean_STD(accTestRunsKNN, KNNNum1)[1];
        System.out.println("KNN Average testing accuracy: " + aveAccTestKNN);
        System.out.println("KNN Standard Deviation of testing accuracy: "
                + stdAccTestKNN);

        // ============================
        // ============================

        System.out.println("");
        double aveTrainAccYan = NewMath.Mean_STD(accTrainRunsYan)[0]; // when
        // Aa=0.0,
        // fitness=
        // trAcc
        double stdTrainAccYan = NewMath.Mean_STD(accTrainRunsYan)[1];
        System.out.println("YAN Average Training accuracy: " + aveTrainAccYan);
        System.out.println("YAN Standard Deviation of Training accuracy:  "
                + stdTrainAccYan);

        System.out.println("");
        double aveTrainAccDT = NewMath.Mean_STD(accTrainRunsDT)[0]; // when
        // Aa=0.0,
        // fitness=
        // trAcc
        double stdTrainAccDT = NewMath.Mean_STD(accTrainRunsDT)[1];
        System.out.println("DT Average Training accuracy: " + aveTrainAccDT);
        System.out.println("DT Standard Deviation of Training accuracy:  "
                + stdTrainAccDT);

        System.out.println("");
        double aveTrainAccNB = NewMath.Mean_STD(accTrainRunsNB)[0]; // when
        // Aa=0.0,
        // fitness=
        // trAcc
        double stdTrainAccNB = NewMath.Mean_STD(accTrainRunsNB)[1];
        System.out.println("NB Average Training accuracy: " + aveTrainAccNB);
        System.out.println("NB Standard Deviation of Training accuracy:  "
                + stdTrainAccNB);

        System.out.println("");
        double aveTrainAccKNN = NewMath.Mean_STD(accTrainRunsKNN)[0]; // when
        // Aa=0.0,
        // fitness=
        // trAcc
        double stdTrainAccKNN = NewMath.Mean_STD(accTrainRunsKNN)[1];
        System.out.println("KNN Average Training accuracy: " + aveTrainAccKNN);
        System.out.println("KNN Standard Deviation of Training accuracy:  "
                + stdTrainAccKNN);

        // =============================
        // =============================

        System.out.println("");
        double aveFitness = NewMath.Mean_STD(bestFitnessRuns)[0]; // when
        // Aa=0.0,
        // fitness=
        // Acc_tr
        double stdFitness = NewMath.Mean_STD(bestFitnessRuns)[1];
        System.out.println("Average fitness : " + aveFitness);
        System.out.println("Standard Deviation of fitness: " + stdFitness);

        System.out.println();
        System.out
                .println("-------------------------------------------------------------------------------------------------");
        System.out
                .println(" Ave-Acc (Best-Acc)      Std-Acc      Ave-TrainAcc      Std-TrainAcc");
        // System.out.println(" &" + df.format(100 * aveAccTest) + " (" +
        // df.format(100 * bestAccTest) + ")"
        // + " &" + dg.format(stdAccTest) + " &" + df.format(100 * aveTrainAcc)
        // + " &" + dg.format(stdTrainAcc));
        //
        // System.out.println("fullSet.Testacc()   " + " & " + df.format(100 *
        // fullTest));

        NumberFormat formatter = new DecimalFormat("0.#####E0");

        System.out.println("================================");
        System.out.println(df.format(aveTrainAccYan * 100));
        System.out.println(df.format(aveAccTestYan * 100));
        System.out.println(df.format(bestAccTest));
        System.out.println(formatter.format(stdAccTestYan));
        System.out.println(formatter.format(stdTrainAccYan));
        System.out.println((int) NewMath.mean(timeRuns));
        System.out.println("================================");

        // CFOrg

        System.out.println("");
        double CFOrg_aveAccTest_Yan = NewMath.Mean_STD(CFOrgAccTestingRunsYan,
                YanNum2)[0];
        double CFOrg_stdAccTest_Yan = NewMath.Mean_STD(CFOrgAccTestingRunsYan,
                YanNum2)[1];
        System.out.println("Yan CFOrg_Average testing accuracy: "
                + CFOrg_aveAccTest_Yan);
        System.out.println("Yan CFOrg_Standard Deviation of testing accuracy: "
                + CFOrg_stdAccTest_Yan);

        System.out.println("");
        double CFOrg_aveAccTest_DT = NewMath.Mean_STD(CFOrgAccTestingRunsDT,
                DTNum2)[0];
        double CFOrg_stdAccTest_DT = NewMath.Mean_STD(CFOrgAccTestingRunsDT,
                DTNum2)[1];
        System.out.println("DT CFOrg_Average testing accuracy: "
                + CFOrg_aveAccTest_DT);
        System.out.println("DT CFOrg_Standard Deviation of testing accuracy: "
                + CFOrg_stdAccTest_DT);

        System.out.println("");
        double CFOrg_aveAccTest_NB = NewMath.Mean_STD(CFOrgAccTestingRunsNB,
                NBNum2)[0];
        double CFOrg_stdAccTest_NB = NewMath.Mean_STD(CFOrgAccTestingRunsNB,
                NBNum2)[1];
        System.out.println("NB CFOrg_Average testing accuracy: "
                + CFOrg_aveAccTest_NB);
        System.out.println("NB CFOrg_Standard Deviation of testing accuracy: "
                + CFOrg_stdAccTest_NB);

        System.out.println("");
        double CFOrg_aveAccTest_KNN = NewMath.Mean_STD(CFOrgAccTestingRunsKNN,
                KNNNum2)[0];
        double CFOrg_stdAccTest_KNN = NewMath.Mean_STD(CFOrgAccTestingRunsKNN,
                KNNNum2)[1];
        System.out.println("KNN CFOrg_Average testing accuracy: "
                + CFOrg_aveAccTest_KNN);
        System.out.println("KNN CFOrg_Standard Deviation of testing accuracy: "
                + CFOrg_stdAccTest_KNN);

        System.out.println("");
        double CFOrg_aveTrainAcc_DT = NewMath.Mean_STD(CFOrgAccTrainingRunsDT)[0]; // when
        // Aa=0.0,
        // fitness=
        // trAcc
        double CFOrg_stdTrainAcc_DT = NewMath.Mean_STD(CFOrgAccTrainingRunsDT)[1];
        System.out.println("DT_CFOrg_Average Training accuracy: "
                + CFOrg_aveTrainAcc_DT);
        System.out
                .println("DT_CFOrg_Standard Deviation of Training accuracy:  "
                        + CFOrg_stdTrainAcc_DT);

        System.out.println("");
        double CFOrg_aveTrainAcc_Yan = NewMath
                .Mean_STD(CFOrgAccTrainingRunsYan)[0]; // when Aa=0.0, fitness=
        // trAcc
        double CFOrg_stdTrainAcc_Yan = NewMath
                .Mean_STD(CFOrgAccTrainingRunsYan)[1];
        System.out.println("Yan_CFOrg_Average Training accuracy: "
                + CFOrg_aveTrainAcc_Yan);
        System.out
                .println("Yan_CFOrg_Standard Deviation of Training accuracy:  "
                        + CFOrg_stdTrainAcc_Yan);

        System.out.println("");
        double CFOrg_aveTrainAcc_NB = NewMath.Mean_STD(CFOrgAccTrainingRunsNB)[0]; // when
        // Aa=0.0,
        // fitness=
        // trAcc
        double CFOrg_stdTrainAcc_NB = NewMath.Mean_STD(CFOrgAccTrainingRunsNB)[1];
        System.out.println("NB_CFOrg_Average Training accuracy: "
                + CFOrg_aveTrainAcc_NB);
        System.out
                .println("NB_CFOrg_Standard Deviation of Training accuracy:  "
                        + CFOrg_stdTrainAcc_NB);

        System.out.println("");
        double CFOrg_aveTrainAcc_KNN = NewMath
                .Mean_STD(CFOrgAccTrainingRunsKNN)[0]; // when Aa=0.0, fitness=
        // trAcc
        double CFOrg_stdTrainAcc_KNN = NewMath
                .Mean_STD(CFOrgAccTrainingRunsKNN)[1];
        System.out.println("KNN_CFOrg_Average Training accuracy: "
                + CFOrg_aveTrainAcc_KNN);
        System.out
                .println("KNN_CFOrg_Standard Deviation of Training accuracy:  "
                        + CFOrg_stdTrainAcc_KNN);
        System.out.println("================================");
        System.out.println(df.format(CFOrg_aveTrainAcc_Yan * 100));
        System.out.println(df.format(CFOrg_aveAccTest_Yan * 100));
        System.out.println();
        System.out.println(formatter.format(CFOrg_stdAccTest_Yan));
        System.out.println(formatter.format(CFOrg_stdTrainAcc_Yan));
        System.out.println("================================");

        double[] averageGbestIterations = new double[number_of_iterations]; // average
        // best
        // fitness
        // of
        // all
        // runs
        // in
        // each
        // iterate
        // for
        // plot
        double[] aveErTrainIterations = new double[number_of_iterations];

        averageGbestIterations = NewMath
                .AverageRunIterations(gbestRunsIterations); // average best
        // fitness of all
        // runs in each
        // iterate
        // averageCaching = NewMath.AverageRunIterations(CachingRunsIterations);
        // // average caching of all runs in each iterate
        aveErTrainIterations = NewMath
                .AverageRunIterations(eroGbestRunsIterations); // average Train
        // Erro of Gbest
        // fitness of
        // all runs in
        // each iterate

        try
        {
            OutputYan out = new OutputYan(dir, fname);
//            out.printYan(number_of_iterations, number_of_runs,
//                    averageGbestIterations, aveErTrainIterations, df, dg,
//                    accTestRunsYan, timeRuns, bestAccTest, aveTrainAccYan,
//                    aveAccTestYan, stdAccTestYan, stdTrainAccYan, fullTest,
//                    aveFitness, stdFitness, accTrainRunsYan, bestFitnessRuns,
//                    bestRun, dimension, bestPositionRuns, gbestRunsIterations,
//                    eroGbestRunsIterations, fullTrain, accTrainRunsYan,
//                    accTestRunsYan, accTrainRunsDT, accTestRunsDT,
//                    CFOrgAccTrainingRunsDT, CFOrgAccTestingRunsDT,
//                    accTrainRunsNB, accTestRunsNB, CFOrgAccTrainingRunsNB,
//                    CFOrgAccTestingRunsNB, accTrainRunsKNN, accTestRunsKNN,
//                    CFOrgAccTrainingRunsKNN, CFOrgAccTestingRunsKNN);

            out.printOperators(operatorsRuns, number_of_runs);
            // CF only
            Dataset fcdata = HelpDataset.removeFeatures(data,
                    bestPositionRuns[bestRun]);
            fcdata = FCFunction.calConstructFeaBing(fcdata, operatorsRuns[bestRun]);
            double[] cf_best_result = null;
            try
            {
                cf_best_result = out.superDataTestingAcc(fcdata, "CF");
            }
            catch (Exception e)
            {
                System.out.println(e.getMessage());
            }
            catch (Error e)
            {
                System.out.println("error here");
            }
            System.out.println();
            System.out.println();

            // CF + org
            double[] cforg_best_result = null;
            try
            {
                cforg_best_result = out.superDataTestingAcc(out
                        .constructNewDataset(data, bestPositionRuns[bestRun],
                                operatorsRuns[bestRun]), "CFOrg");
            }
            catch (Exception e)
            {
                System.out.println(e.getMessage());
            }
            catch (Error e)
            {
                System.out.println("error here");
            }

            double[] orgTest = OrgClassification.excuteClassification2(fname);

            double orgDT = orgTest[0];
            double orgKNN = orgTest[1];
            double orgNB = orgTest[2];
            System.out.println(orgDT + " " + orgKNN + " " + orgNB);

            double orgDTtr = orgTest[3];
            double orgKNNtr = orgTest[4];
            double orgNBtr = orgTest[5];

            /**
             * This is for the indiviual checking
             */
            LatexFormat.dir = dir;
            LatexFormat.printClassifier("DT", fname, noFeatures, df, dg,
                    accTestRunsDT, aveAccTestDT, stdAccTestDT, accTrainRunsDT,
                    aveTrainAccDT, stdTrainAccDT, CFOrgAccTestingRunsDT,
                    CFOrg_aveAccTest_DT, CFOrg_stdAccTest_DT,
                    CFOrgAccTrainingRunsDT, CFOrg_aveTrainAcc_DT,
                    CFOrg_stdTrainAcc_DT, orgDT, orgDTtr);

            LatexFormat.printClassifier("KNN", fname, noFeatures, df, dg,
                    accTestRunsKNN, aveAccTestKNN, stdAccTestKNN,
                    accTrainRunsKNN, aveTrainAccKNN, stdTrainAccKNN,
                    CFOrgAccTestingRunsKNN, CFOrg_aveAccTest_KNN,
                    CFOrg_stdAccTest_KNN, CFOrgAccTrainingRunsKNN,
                    CFOrg_aveTrainAcc_KNN, CFOrg_stdTrainAcc_KNN, orgKNN, orgKNNtr);

            LatexFormat.printClassifier("NB", fname, noFeatures, df, dg,
                    accTestRunsNB, aveAccTestNB, stdAccTestNB, accTrainRunsNB,
                    aveTrainAccNB, stdTrainAccNB, CFOrgAccTestingRunsNB,
                    CFOrg_aveAccTest_NB, CFOrg_stdAccTest_NB,
                    CFOrgAccTrainingRunsNB, CFOrg_aveTrainAcc_NB,
                    CFOrg_stdTrainAcc_NB, orgNB, orgNBtr);

            System.out
                    .println("=================================================");
            System.out
                    .println("=================================================");

            /**
             * This is for Latex
             */

            LatexFormat.printBigLatexTable(fname, noFeatures, df, dg, orgDT,
                    orgKNN, orgNB, accTestRunsDT, aveAccTestDT, stdAccTestDT,
                    accTestRunsKNN, aveAccTestKNN, stdAccTestKNN,
                    accTestRunsNB, aveAccTestNB, stdAccTestNB,
                    CFOrgAccTestingRunsDT, CFOrg_aveAccTest_DT,
                    CFOrg_stdAccTest_DT, CFOrgAccTestingRunsKNN,
                    CFOrg_aveAccTest_KNN, CFOrg_stdAccTest_KNN,
                    CFOrgAccTestingRunsNB, CFOrg_aveAccTest_NB,
                    CFOrg_stdAccTest_NB);
        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        System.out.println();
        System.out.println("====================================");
        System.out.println("y1, y2: " + YanNum1 + " " + YanNum2);
        System.out.println("knn1, knn2: " + KNNNum1 + " " + KNNNum2);
        System.out.println("DT1, DT2: " + DTNum1 + " " + DTNum2);
        System.out.println("NB1, NB2: " + NBNum1 + " " + NBNum2);
    }
}
