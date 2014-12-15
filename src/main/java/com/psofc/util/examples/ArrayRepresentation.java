package com.psofc.util.examples;

import com.psofc.*;
import com.psofc.singlepsofc.arrayrep.FeatureConstruction;
import com.psofc.singlepsofc.arrayrep.OperatorSelection;
import com.psofc.util.*;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;

import java.text.DecimalFormat;
import java.text.NumberFormat;
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

            // summarise a run
            recordIteration(r, arrRecord, initTime);
        } // end all run

        /**
         * start summarise all run
         *
         * get best testing accuracy and its feature size If two jobs produce
         * the same accuracies, the one with smaller size will be regarded as
         * the better one ;
         */

        int bestRun = getBestRun(arrRecord);


    }

    private void printResult(ArrRecord arrRecord, int bestRun)
    {
        System.out.println("Fitness                 TrainAcc            Acc_Test  (in each run)");
        for (int r = 0; r < number_of_runs; ++r)
        {
            // System.out.println("The" + (j + 1) + "Run");
            System.out.println(arrRecord.bestFitnessRuns[r] + "   " + arrRecord.accTrainRunsYan[r] + "          " + arrRecord.accTestRunsYan[r]);
        }

        System.out.println("");
        System.out.println("Job for the Best testing acc is:  " + (bestRun + 1)
                + " Run (index+1) ");
        System.out.println("position for the Best testing accuracy is: ");

        int o = 0;
        char[] operators = OperatorSelection.getOpFromArray(arrRecord.bestPositionRuns[bestRun]);
        for (int x = 0; x < operators.length; x++)
        {
            System.out.print(operators[x] + " ");
        }
        System.out.println();
        for (int d = 0; d < dimension; d++)
        {
            if (arrRecord.bestPositionRuns[bestRun][d] >= 0.5 && d % 2 == 0)
                System.out.print(arrRecord.bestPositionRuns[bestRun][d] + " ( f"
                        + (d / 2) + " ) " + operators[o++] + " ");
        }

        System.out.println();
        System.out.println("feature used: " + (o));
        System.out.println("");
        System.out.println("Best testing accuracy: " + arrRecord.bestAccTest);

        System.out.println("");
        double aveAccTestYan = NewMath.Mean_STD(arrRecord.accTestRunsYan, arrRecord.YanNum1)[0];
        double stdAccTestYan = NewMath.Mean_STD(arrRecord.accTestRunsYan, arrRecord.YanNum1)[1];
        System.out.println("Yan Average testing accuracy: " + aveAccTestYan);
        System.out.println("Yan Standard Deviation of testing accuracy: "
                + stdAccTestYan);

        System.out.println("");
        double aveAccTestDT = NewMath.Mean_STD(arrRecord.accTestRunsDT, arrRecord.DTNum1)[0];
        double stdAccTestDT = NewMath.Mean_STD(arrRecord.accTestRunsDT, arrRecord.DTNum1)[1];
        System.out.println("DT Average testing accuracy: " + aveAccTestDT);
        System.out.println("DT Standard Deviation of testing accuracy: "
                + stdAccTestDT);

        System.out.println("");
        double aveAccTestNB = NewMath.Mean_STD(arrRecord.accTestRunsNB, arrRecord.NBNum1)[0];
        double stdAccTestNB = NewMath.Mean_STD(arrRecord.accTestRunsNB, arrRecord.NBNum1)[1];
        System.out.println("NB Average testing accuracy: " + aveAccTestNB);
        System.out.println("NB Standard Deviation of testing accuracy: "
                + stdAccTestNB);

        System.out.println("");
        double aveAccTestKNN = NewMath.Mean_STD(arrRecord.accTestRunsKNN, arrRecord.KNNNum1)[0];
        double stdAccTestKNN = NewMath.Mean_STD(arrRecord.accTestRunsKNN, arrRecord.KNNNum1)[1];
        System.out.println("KNN Average testing accuracy: " + aveAccTestKNN);
        System.out.println("KNN Standard Deviation of testing accuracy: "
                + stdAccTestKNN);


        // ============================
        //
        // ============================

        System.out.println("");
        double aveTrainAccYan = NewMath.Mean_STD(arrRecord.accTrainRunsYan)[0]; // when
        // Aa=0.0, fitness=trAcc
        double stdTrainAccYan = NewMath.Mean_STD(arrRecord.accTrainRunsYan)[1];
        System.out.println("YAN Average Training accuracy: " + aveTrainAccYan);
        System.out.println("YAN Standard Deviation of Training accuracy:  " + stdTrainAccYan);

        System.out.println("");
        double aveTrainAccDT = NewMath.Mean_STD(arrRecord.accTrainRunsDT)[0]; // when Aa=0.0, fitness= trAcc
        double stdTrainAccDT = NewMath.Mean_STD(arrRecord.accTrainRunsDT)[1];
        System.out.println("DT Average Training accuracy: " + aveTrainAccDT);
        System.out.println("DT Standard Deviation of Training accuracy:  " + stdTrainAccDT);

        System.out.println("");
        double aveTrainAccNB = NewMath.Mean_STD(arrRecord.accTrainRunsNB)[0]; // when Aa=0.0, fitness= trAcc
        double stdTrainAccNB = NewMath.Mean_STD(arrRecord.accTrainRunsNB)[1];
        System.out.println("NB Average Training accuracy: " + aveTrainAccNB);
        System.out.println("NB Standard Deviation of Training accuracy:  " + stdTrainAccNB);

        System.out.println("");
        double aveTrainAccKNN = NewMath.Mean_STD(arrRecord.accTrainRunsKNN)[0]; // when Aa=0.0, fitness= trAcc
        double stdTrainAccKNN = NewMath.Mean_STD(arrRecord.accTrainRunsKNN)[1];
        System.out.println("KNN Average Training accuracy: " + aveTrainAccKNN);
        System.out.println("KNN Standard Deviation of Training accuracy:  " + stdTrainAccKNN);


        // =============================
        // =============================

        System.out.println("");
        double aveFitness = NewMath.Mean_STD(arrRecord.bestFitnessRuns)[0]; // when
        // Aa=0.0, fitness= Acc_tr
        double stdFitness = NewMath.Mean_STD(arrRecord.bestFitnessRuns)[1];
        System.out.println("Average fitness : " + aveFitness);
        System.out.println("Standard Deviation of fitness: " + stdFitness);

        System.out.println();
        System.out
                .println("-------------------------------------------------------------------------------------------------");
        System.out
                .println(" Ave-Acc (Best-Acc)      Std-Acc      Ave-TrainAcc      Std-TrainAcc");
        System.out.println(" &" + df.format(100 * aveAccTestYan) + " ("
                + df.format(100 * arrRecord.bestAccTest) + ")" + " &"
                + dg.format(stdAccTestYan) + " &"
                + df.format(100 * aveTrainAccYan) + " &"
                + dg.format(stdTrainAccYan));

        System.out.println("fullSet.Testacc()   " + " & "
                + df.format(100 * fullTest));

        NumberFormat formatter = new DecimalFormat("0.#####E0");

        System.out.println("=============CF only=================");
        System.out.println(((int) (aveTrainAccYan * 10000)) / 100.0);
        System.out.println(((int) (aveAccTestYan * 10000)) / 100.0);
        System.out.println(((int) (arrRecord.bestAccTest * 10000)) / 100.0);
        System.out.println(formatter.format(stdAccTestYan)); // DT
        System.out.println(formatter.format(stdTrainAccYan)); // DT
        System.out.println((int) NewMath.mean(arrRecord.timeRuns));
        System.out.println("================================");


        // CFOrg

        System.out.println("");
        double CFOrg_aveAccTest_Yan = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsYan, arrRecord.YanNum2)[0];
        double CFOrg_stdAccTest_Yan = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsYan, arrRecord.YanNum2)[1];
        System.out.println("Yan CFOrg_Average testing accuracy: "
                + CFOrg_aveAccTest_Yan);
        System.out.println("Yan CFOrg_Standard Deviation of testing accuracy: "
                + CFOrg_stdAccTest_Yan);

        System.out.println("");
        double CFOrg_aveAccTest_DT = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsDT, arrRecord.DTNum2)[0];
        double CFOrg_stdAccTest_DT = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsDT, arrRecord.DTNum2)[1];
        System.out.println("DT CFOrg_Average testing accuracy: " + CFOrg_aveAccTest_DT);
        System.out.println("DT CFOrg_Standard Deviation of testing accuracy: " + CFOrg_stdAccTest_DT);

        System.out.println("");
        double CFOrg_aveAccTest_NB = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsNB, arrRecord.NBNum2)[0];
        double CFOrg_stdAccTest_NB = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsNB, arrRecord.NBNum2)[1];
        System.out.println("NB CFOrg_Average testing accuracy: " + CFOrg_aveAccTest_NB);
        System.out.println("NB CFOrg_Standard Deviation of testing accuracy: " + CFOrg_stdAccTest_NB);

        System.out.println("");
        double CFOrg_aveAccTest_KNN = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsKNN, arrRecord.KNNNum2)[0];
        double CFOrg_stdAccTest_KNN = NewMath.Mean_STD(arrRecord.CFOrgAccTestingRunsKNN, arrRecord.KNNNum2)[1];
        System.out.println("KNN CFOrg_Average testing accuracy: " + CFOrg_aveAccTest_KNN);
        System.out.println("KNN CFOrg_Standard Deviation of testing accuracy: " + CFOrg_stdAccTest_KNN);

        System.out.println("");
        double CFOrg_aveTrainAcc_DT = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsDT)[0]; // when
        // Aa=0.0, fitness= trAcc
        double CFOrg_stdTrainAcc_DT = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsDT)[1];
        System.out.println("DT_CFOrg_Average Training accuracy: " + CFOrg_aveTrainAcc_DT);
        System.out
                .println("DT_CFOrg_Standard Deviation of Training accuracy:  " + CFOrg_stdTrainAcc_DT);

        System.out.println("");
        double CFOrg_aveTrainAcc_Yan = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsYan)[0]; // when Aa=0.0, fitness= trAcc
        double CFOrg_stdTrainAcc_Yan = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsYan)[1];
        System.out.println("Yan_CFOrg_Average Training accuracy: " + CFOrg_aveTrainAcc_Yan);
        System.out.println("Yan_CFOrg_Standard Deviation of Training accuracy:  " + CFOrg_stdTrainAcc_Yan);

        System.out.println("");
        double CFOrg_aveTrainAcc_NB = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsNB)[0]; // when Aa=0.0, fitness= trAcc
        double CFOrg_stdTrainAcc_NB = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsNB)[1];
        System.out.println("NB_CFOrg_Average Training accuracy: " + CFOrg_aveTrainAcc_NB);
        System.out
                .println("NB_CFOrg_Standard Deviation of Training accuracy:  " + CFOrg_stdTrainAcc_NB);

        System.out.println("");
        double CFOrg_aveTrainAcc_KNN = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsKNN)[0]; // when Aa=0.0, fitness=
        // trAcc
        double CFOrg_stdTrainAcc_KNN = NewMath.Mean_STD(arrRecord.CFOrgAccTrainingRunsKNN)[1];
        System.out.println("KNN_CFOrg_Average Training accuracy: " + CFOrg_aveTrainAcc_KNN);
        System.out.println("KNN_CFOrg_Standard Deviation of Training accuracy:  " + CFOrg_stdTrainAcc_KNN);

        System.out.println("================CFOrg===============");
        System.out.println((((int) (CFOrg_aveTrainAcc_DT * 10000)) / 100.0));
        System.out.println(((int) (CFOrg_aveAccTest_DT * 10000)) / 100.0);
        System.out.println();
        System.out.println(formatter.format(CFOrg_stdAccTest_DT)); // DT
        System.out.println(formatter.format(CFOrg_stdTrainAcc_DT)); // DT
        System.out.println("================================");

        double[] averageGbestIterations = new double[number_of_iterations]; // average best fitness of all runs in each iterate for plot
        double[] aveErTrainIterations = new double[number_of_iterations];

        // average best fitness of all runs in each iterate
        averageGbestIterations = NewMath.AverageRunIterations(arrRecord.gbestRunsIterations);

        // averageCaching = NewMath.AverageRunIterations(CachingRunsIterations);
        // average caching of all runs in each iterate
        aveErTrainIterations = NewMath.AverageRunIterations(arrRecord.eroGbestRunsIterations);// average Train
        // Error of Gbest fitness of all runs in each iterate


        try
        {
            OutputYan out = new OutputYan(dir, fname);
//            out.printYan(number_of_iterations, number_of_runs,
//                    averageGbestIterations, aveErTrainIterations,
//                    df, dg, aveTrainAccYan,
//                    aveAccTestYan, stdAccTestYan, stdTrainAccYan,
//                    fullTest, aveFitness, stdFitness,
//                    bestRun, dimension, fullTrain, ArrRecord);

            out.printOperators(arrRecord.operatorsRuns, number_of_runs);

            out.printConstructedFeature(arrRecord.constructedFeatureTrRuns, arrRecord.constructedFeatureTtRuns, number_of_runs);

            // CF only

            double[] features = new double[(arrRecord.bestPositionRuns[bestRun].length + 1) / 2];
            int p = 0;
            for (int i = 0; i < arrRecord.bestPositionRuns[bestRun].length; i += 2)
            {

                features[p++] = arrRecord.bestPositionRuns[bestRun][i];
            }

            Dataset fcdata = HelpDataset.removeFeatures(data, features);
            fcdata = FCFunction.calConstructFeaBing(fcdata, arrRecord.operatorsRuns[bestRun]);
            double[] cf_best_result = null;
            try
            {
                cf_best_result = out
                        .superDataTestingAcc(fcdata, "CF");
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }
            System.out.println();
            System.out.println();

            // CF + org
            double[] cforg_best_result = null;
            try
            {
                cforg_best_result = out.superDataTestingAcc(
                        out.constructNewDataset(data, features, arrRecord.operatorsRuns[bestRun]), "CFOrg");
            }
            catch (Exception e)
            {
                e.printStackTrace();
            }

            double[] orgTest = OrgClassification.excuteClassification2(fname);

            double orgDT = orgTest[0];
            double orgKNN = orgTest[1];
            double orgNB = orgTest[2];
            System.out.println(orgDT + " " + orgKNN + " " + orgNB);
//            double orgDTtr = orgTest[3];
//            double orgKNNtr = orgTest[4];
//            double orgNBtr = orgTest[5];

//            /**
//             * This is for the indiviual checking
//             */
//
//            LatexFormat.dir = dir;
//
//            LatexFormat.printClassifier("DT", fname, noFeatures, df, dg,
//                    accTestRunsDT, aveAccTestDT, stdAccTestDT, accTrainRunsDT,
//                    aveTrainAccDT, stdTrainAccDT, CFOrgAccTestingRunsDT,
//                    CFOrg_aveAccTest_DT, CFOrg_stdAccTest_DT,
//                    CFOrgAccTrainingRunsDT, CFOrg_aveTrainAcc_DT,
//                    CFOrg_stdTrainAcc_DT, orgDT, orgDTtr);
//
//            LatexFormat.printClassifier("KNN", fname, noFeatures, df, dg,
//                    accTestRunsKNN, aveAccTestKNN, stdAccTestKNN,
//                    accTrainRunsKNN, aveTrainAccKNN, stdTrainAccKNN,
//                    CFOrgAccTestingRunsKNN, CFOrg_aveAccTest_KNN,
//                    CFOrg_stdAccTest_KNN, CFOrgAccTrainingRunsKNN,
//                    CFOrg_aveTrainAcc_KNN, CFOrg_stdTrainAcc_KNN, orgKNN, orgKNNtr);
//
//            LatexFormat.printClassifier("NB", fname, noFeatures, df, dg,
//                    accTestRunsNB, aveAccTestNB, stdAccTestNB, accTrainRunsNB,
//                    aveTrainAccNB, stdTrainAccNB, CFOrgAccTestingRunsNB,
//                    CFOrg_aveAccTest_NB, CFOrg_stdAccTest_NB,
//                    CFOrgAccTrainingRunsNB, CFOrg_aveTrainAcc_NB,
//                    CFOrg_stdTrainAcc_NB, orgNB, orgNBtr);
//
//            System.out
//                    .println("=================================================");
//            System.out
//                    .println("=================================================");
//
//            /**
//             * This is for Latex
//             */
//
//            LatexFormat.printBigLatexTable(fname, noFeatures, df, dg,
//                    orgDT, orgKNN, orgNB,
//                    accTestRunsDT, aveAccTestDT, stdAccTestDT,
//                    accTestRunsKNN, aveAccTestKNN, stdAccTestKNN,
//                    accTestRunsNB, aveAccTestNB, stdAccTestNB,
//                    CFOrgAccTestingRunsDT, CFOrg_aveAccTest_DT, CFOrg_stdAccTest_DT,
//                    CFOrgAccTestingRunsKNN, CFOrg_aveAccTest_KNN, CFOrg_stdAccTest_KNN,
//                    CFOrgAccTestingRunsNB, CFOrg_aveAccTest_NB, CFOrg_stdAccTest_NB);


        }
        catch (Exception e)
        {
            e.printStackTrace();
        }

        System.out.println();
        System.out.println("====================================");
        System.out.println("y1, y2: " + arrRecord.YanNum1 + " " + arrRecord.YanNum2);
        System.out.println("knn1, knn2: " + arrRecord.KNNNum1 + " " + arrRecord.KNNNum2);
        System.out.println("DT1, DT2: " + arrRecord.DTNum1 + " " + arrRecord.DTNum2);
        System.out.println("NB1, NB2: " + arrRecord.NBNum1 + " " + arrRecord.NBNum2);



    }

    private int getBestRun(ArrRecord arrRecord)
    {
        // get best run
        int bestRun = 0;
        for (int r = 0; r < number_of_runs; r++)
        {
            if (arrRecord.accTestRunsYan[r] > arrRecord.bestAccTest)
            {
                arrRecord.bestAccTest = arrRecord.accTestRunsYan[r];
                bestRun = r;
            }
        }
        return bestRun;
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
        double bestAccTest = 0.0;

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


    public static void main(String[] arg)
    {
        String fname = "australian";
        String dir = "file_array";
        BaseRepresentationRun br = new ArrayRepresentation();

        if(br.setup(fname, dir))
        {
            br.start(br.getRandomSeeder());
        }
    }

}
