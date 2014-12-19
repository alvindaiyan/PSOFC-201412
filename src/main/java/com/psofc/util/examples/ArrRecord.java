package com.psofc.util.examples;

/**
 * Created by daiyan on 19/12/14.
 */

public class ArrRecord implements PrintResult.Record
{


    public ArrRecord(int r, int number_of_iterations, int dimension, int training_size, int testing_size)
    {
        this.number_of_runs = r;
        this.number_of_iterations = number_of_iterations;
        this.dimension = dimension;
        this.training_size = training_size;
        this.testing_size = testing_size;


        bestAccTest = 0.0;

        gbestRunsIterations = new double[number_of_runs][number_of_iterations];
        eroGbestRunsIterations = new double[number_of_runs][number_of_iterations];

        bestFitnessRuns = new double[number_of_runs];

        accTestRunsYan = new double[number_of_runs];
        accTrainRunsYan = new double[number_of_runs];
        YanNum1 = number_of_runs;
        YanNum2 = number_of_runs;

        accTestRunsDT = new double[number_of_runs];
        accTrainRunsDT = new double[number_of_runs];
        DTNum1 = number_of_runs;
        DTNum2 = number_of_runs;

        accTestRunsKNN = new double[number_of_runs];
        accTrainRunsKNN = new double[number_of_runs];
        KNNNum1 = number_of_runs;
        KNNNum2 = number_of_runs;

        accTestRunsNB = new double[number_of_runs];
        accTrainRunsNB = new double[number_of_runs];
        NBNum1 = number_of_runs;
        NBNum2 = number_of_runs;

        bestPositionRuns = new double[number_of_runs][dimension];

        timeRuns = new long[number_of_runs];

        CFOrgAccTestingRunsDT = new double[number_of_runs];
        CFOrgAccTrainingRunsDT = new double[number_of_runs];

        CFOrgAccTestingRunsKNN = new double[number_of_runs];
        CFOrgAccTrainingRunsKNN = new double[number_of_runs];

        CFOrgAccTestingRunsNB = new double[number_of_runs];
        CFOrgAccTrainingRunsNB = new double[number_of_runs];

        CFOrgAccTestingRunsYan = new double[number_of_runs];
        CFOrgAccTrainingRunsYan = new double[number_of_runs];

        operatorsRuns = new char[number_of_runs][dimension - 1];

        constructedFeatureTrRuns = new double[number_of_runs][training_size];
        constructedFeatureTtRuns = new double[number_of_runs][testing_size];
    }

    private int number_of_runs;
    private int number_of_iterations;
    private int dimension;
    private int training_size;
    private int testing_size;

    protected double bestAccTest = 0.0;
    protected double[][] gbestRunsIterations; // get best fitness in each iterate in each run
    protected double[][] eroGbestRunsIterations;
    protected double[] bestFitnessRuns; // get final bestfitnes in each run
    protected double[] accTestRunsYan; // the best testing accuracy in each run
    protected double[] accTrainRunsYan;
    protected int YanNum1 = number_of_runs;
    protected int YanNum2 = number_of_runs;
    protected double[] accTestRunsDT; // the best testing accuracy in each run
    protected double[] accTrainRunsDT;
    protected int DTNum1 = number_of_runs;
    protected int DTNum2 = number_of_runs;
    protected double[] accTestRunsKNN; // the best testing accuracy in each run
    protected double[] accTrainRunsKNN;
    protected int KNNNum1;
    protected int KNNNum2;
    protected double[] accTestRunsNB; // the best testing accuracy in each run
    protected double[] accTrainRunsNB;
    protected int NBNum1;
    protected int NBNum2;
    protected double[][] bestPositionRuns;// get the position of best results in each run;
    protected long[] timeRuns;
    protected double[] CFOrgAccTestingRunsDT;
    protected double[] CFOrgAccTrainingRunsDT;
    protected double[] CFOrgAccTestingRunsKNN;
    protected double[] CFOrgAccTrainingRunsKNN;
    protected double[] CFOrgAccTestingRunsNB;
    protected double[] CFOrgAccTrainingRunsNB;
    protected double[] CFOrgAccTestingRunsYan;
    protected double[] CFOrgAccTrainingRunsYan;
    protected char[][] operatorsRuns; // temperory, this need to according the results of the selected operators
    protected double[][] constructedFeatureTrRuns;
    protected double[][] constructedFeatureTtRuns;



}