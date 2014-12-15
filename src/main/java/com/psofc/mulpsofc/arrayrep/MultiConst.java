package com.psofc.mulpsofc.arrayrep;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

import com.psofc.singlepsofc.arrayrep.OperatorSelection;
import com.psofc.HelpDataset;
import com.psofc.MyClassifier;
import com.psofc.Particle;
import com.psofc.RandomBing;
import com.psofc.Swarm;
import com.psofc.Topology;
import com.psofc.TopologyRing;
import com.psofc.VelocityClampBasic;
import com.psofc.util.FCFunction;
import com.psofc.util.LatexFormat;
import com.psofc.util.NewMath;
import com.psofc.util.OrgClassification;
import com.psofc.util.ReadResults;


import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.tools.data.FileHandler;

public class MultiConst {
//	int changablevalue = 0;

	int number_of_runs = 50;
	double w = 0.729844;
	double c1 = 1.49618, c2 = 1.49618;
	int number_of_particles = 30;
	int number_of_iterations = 100;
	Topology topology = new TopologyRing(30);
	int numFolds = 10;

	DecimalFormat df = new DecimalFormat("##.##");
	DecimalFormat dg = new DecimalFormat("##.#E0");

	public MultiConst(String fn) throws IOException{

		String fname = fn;
		String dir = "file_multi_array";



		int noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname
				+ "/noFeatures.txt"));
		Dataset data = FileHandler.loadDataset(new File("Data/" + fname
				+ "/Data.data"), noFeatures, ",");
		int dimension = noFeatures * 2 - 1;


		MulFeatureConstruction problem = new MulFeatureConstruction();



//		problem.setMyclassifier(new MyClassifier(new Random(1))); // new
//																	// ClassifierKNN()
//		problem.getMyclassifier().ClassifierDT();



		/**
		 * set up the training and testing data
		 */
		System.out.println("multi feature construction" + fname + " with class number: " + data.classes().size());

		Dataset[] folds = data.folds((numFolds), new Random(100));
		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();
		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
//		int[] tr = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
//		int[] te = {4}; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			testing.addAll(folds[te[i]]);
		}
		problem.setMyclassifier(new MyClassifier(new Random(1)));
		problem.getMyclassifier().ClassifierDT();

//		double fullTrain = problem.getMyclassifier().fullclassify(training,
//				training);
//		double fullTest = problem.getMyclassifier().fullclassify(training,
//				testing);

		/** Sava the results */
		double[][][] gbestRunsIterations = new double[number_of_runs][data.classes().size()][number_of_iterations]; // get best fitness in each iterate in each run
		double[][][] eroGbestRunsIterations = new double[number_of_runs][data.classes().size()][number_of_iterations];

		double[][] bestFitnessRuns = new double[number_of_runs][data.classes().size()]; // get final bestfitnes in each run


		double[][] accTrainRunsYan = new double[number_of_runs][data.classes().size()];


		double[][][] bestPositionRuns = new double[number_of_runs][data.classes().size()][dimension];// get the position of best results in each run;



		char[][][] operatorsRuns = new char[number_of_runs][data.classes().size()][dimension - 1]; // temperory, this need to
													// according the results of
													// the selected operators




		/*
		 * all the result needed
		 */


		double[] accTestRunsDT = new double[number_of_runs]; // the best testing accuracy in each run
		double[] accTrainRunsDT = new double[number_of_runs];


		double[] accTestRunsKNN = new double[number_of_runs]; // the best testing accuracy in each run
		double[] accTrainRunsKNN = new double[number_of_runs];


		double[] accTestRunsNB = new double[number_of_runs]; // the best testing accuracy in each run
		double[] accTrainRunsNB = new double[number_of_runs];




		double[] CFOrgAccTestingRunsDT = new double[number_of_runs];
		double[] CFOrgAccTrainingRunsDT = new double[number_of_runs];

		double[] CFOrgAccTestingRunsKNN = new double[number_of_runs];
		double[] CFOrgAccTrainingRunsKNN = new double[number_of_runs];

		double[] CFOrgAccTestingRunsNB = new double[number_of_runs];
		double[] CFOrgAccTrainingRunsNB = new double[number_of_runs];


		long[] timeRuns = new long[number_of_runs];


		/** Produce the random seeder */
		long[] Seeder = new long[number_of_runs];
		for (int r = 0; r < number_of_runs; r++) {
			Seeder[r] = r * r * r * 135 + r * r * 246 + 78;
		}
		Dataset[] foldsTrain = training.folds(numFolds, new Random(1));
		/** start PSO */
//		double previousGB = 0.0; //TODO
		for (int r = 0; r < number_of_runs; ++r) {
			System.out.println("**********************************************************************************");
			long initTime = System.currentTimeMillis();
			List<ConstructedFeature> constructedfeatures = new ArrayList<ConstructedFeature>();
			for(Object clz:data.classes()){
				System.out.println("=============class: " + clz + " =================");;

				RandomBing.Seeder.setSeed(Seeder[r]);

				System.out.println("RandomBing.Seeder in  " + (r + 1)
						+ " run is:  " + Seeder[r]);

				Swarm s = new Swarm();
				problem.setClz(clz);
				s.setProblem(problem);
				s.getProblem().setTraining(training); // to set the classes and the
														// attributs in feature
														// selection();
				s.getProblem().setNumFolds(numFolds);
				s.getProblem().setFoldsTrain(foldsTrain);
				s.getProblem().setThreshold(0.6);
				s.setTopology(topology);
				s.setVelocityClamp(new VelocityClampBasic());

				for (int i = 0; i < number_of_particles; ++i) {
					Particle p = new Particle();
					p.setProblem(s.getProblem());
					p.setSize(dimension);
					p.setC1(c1);
					p.setC2(c2);
					s.addParticle(p);

				}
				s.initialize();

				for(int i = 0; i < number_of_iterations; ++i){

					s.iterate(w);

//					if(changablevalue >= this.number_of_iterations * 0.4){
//						s.initialize();
//						changablevalue = 0;
//					}

					double bestFitness = s.getParticle(0).getNeighborhoodFitness();

					System.out.println(bestFitness);

//					if(previousGB == bestFitness){changablevalue++;}
//					previousGB = bestFitness;

					int bestParticle = 0;

					for (int p = 0; p < number_of_particles; p++) {
						if (s.getProblem().isBetter(
								s.getParticle(p).getNeighborhoodFitness(),
								bestFitness)) {
							bestFitness = s.getParticle(p).getNeighborhoodFitness();
							bestParticle = p;
						}
					}

					gbestRunsIterations[r][constructedfeatures.size()][i] = bestFitness;

					eroGbestRunsIterations[r][constructedfeatures.size()][i] = bestFitness; // TODO: fitness = error;

					/*
					 * / get the position of best results in the last generation in
					 * each run; to get the best solution for best fitness in each
					 * run
					 */
					if (i == number_of_iterations - 1) {
						bestFitnessRuns[r][constructedfeatures.size()] = gbestRunsIterations[r][constructedfeatures.size()][i];
						accTrainRunsYan[r][constructedfeatures.size()] = 1.0 - eroGbestRunsIterations[r][constructedfeatures.size()][i];
						// System.arraycopy(s.getParticle(bestParticle).getNeighborhoodPosition().toArray(),
						// 0, bestPositionRuns[r], 0, dimension);
						for (int d = 0; d < s.getParticle(bestParticle)
								.getNeighborhoodPosition().size(); d++) {
							bestPositionRuns[r][constructedfeatures.size()][d] = s.getParticle(bestParticle)
									.getNeighborhoodPosition(d);
						}
					}

				} // end all iterations


				operatorsRuns[r][constructedfeatures.size()] = OperatorSelection.getOpFromArray(bestPositionRuns[r][constructedfeatures.size()]);

				double[] features = NewMath.Position2FeaturesArray(bestPositionRuns[r][constructedfeatures.size()]);

				constructedfeatures.add(new ConstructedFeature(operatorsRuns[r][constructedfeatures.size()], features));

			} // end of all classes

			long estimatedTime = System.currentTimeMillis() - initTime;

			timeRuns[r] = estimatedTime;

			System.out.println("estimatedTime: " + estimatedTime);

			List<Dataset> dtrs = new ArrayList<Dataset>();
			List<Dataset> dtts = new ArrayList<Dataset>();


			int inx = 0;
			for(ConstructedFeature cf:constructedfeatures){
				System.out.println("feature " + inx++ + ":" );
				Dataset temTrain = training.copy();
				Dataset temTest = testing.copy();

				temTrain = HelpDataset.removeFeatures(temTrain, cf.features);
				temTrain = FCFunction.calConstructFeaBing(temTrain, cf.operators);

				dtrs.add(temTrain);


				for(Object tcz:data.classes()){
					System.out.println(problem.findInterval(temTrain, tcz));
					System.out.println(problem.testFitness(temTrain, tcz));
				}

				temTest = HelpDataset.removeFeatures(temTest, cf.features);
				temTest = FCFunction.calConstructFeaBing(temTest, cf.operators);

				dtts.add(temTest);
			}


			Dataset multTrain = FCFunction.constrMulDataset(dtrs);
			Dataset multTest = FCFunction.constrMulDataset(dtts);



			int dataSetSize = noFeatures + data.classes().size();


			Dataset cforgTr = FCFunction.constrMulOrgDataset(training.copy(), multTrain, dataSetSize, noFeatures);
			Dataset cforgTt = FCFunction.constrMulOrgDataset(testing.copy(), multTest, dataSetSize, noFeatures);

			MyClassifier mc = new MyClassifier(new Random(1));
			mc.ClassifierDT();

			accTestRunsDT[r] = mc.classify(multTrain, multTest);
			System.err.println("runs at " + r + ": " + accTestRunsDT[r]);
			accTrainRunsDT[r] = mc.classify(multTrain, multTrain);
			System.err.println("runs at " + r + ": " + accTrainRunsDT[r]);

			CFOrgAccTestingRunsDT[r] = mc.classify(cforgTr, cforgTt);
			System.err.println("cforgtt runs at " + r + ": " + CFOrgAccTestingRunsDT[r]);
			CFOrgAccTrainingRunsDT[r] = mc.classify(cforgTr, cforgTr);
			System.err.println("cforgtr runs at " + r + ": " + CFOrgAccTrainingRunsDT[r]);


			mc.ClassifierKNN();
			accTestRunsKNN[r] = mc.classify(multTrain,  multTest);
			accTrainRunsKNN[r] = mc.classify(multTrain, multTrain);

			CFOrgAccTestingRunsKNN[r] = mc.classify(cforgTr, cforgTt);
			CFOrgAccTrainingRunsKNN[r] = mc.classify(cforgTr, cforgTr);

			mc.ClassifierNB();
			accTestRunsNB[r] = mc.classify(multTrain, multTest);
			accTrainRunsNB[r] = mc.classify(multTrain, multTrain);

			CFOrgAccTestingRunsNB[r] = mc.classify(cforgTr, cforgTt);
			CFOrgAccTrainingRunsNB[r] = mc.classify(cforgTr, cforgTr);

		} // end of all runs

		System.out.println("end of the program");


		System.out.println("");
		double aveAccTestDT = NewMath.Mean_STD(accTestRunsDT, accTestRunsDT.length)[0];
		double stdAccTestDT = NewMath.Mean_STD(accTestRunsDT, accTestRunsDT.length)[1];
		System.out.println("DT Average testing accuracy: " + aveAccTestDT);
		System.out.println("DT Standard Deviation of testing accuracy: "
				+ stdAccTestDT);

		System.out.println("");
		double aveAccTestNB = NewMath.Mean_STD(accTestRunsNB, accTestRunsNB.length)[0];
		double stdAccTestNB = NewMath.Mean_STD(accTestRunsNB, accTestRunsNB.length)[1];
		System.out.println("NB Average testing accuracy: " + aveAccTestNB);
		System.out.println("NB Standard Deviation of testing accuracy: "
				+ stdAccTestNB);

		System.out.println("");
		double aveAccTestKNN = NewMath.Mean_STD(accTestRunsKNN, accTestRunsKNN.length)[0];
		double stdAccTestKNN = NewMath.Mean_STD(accTestRunsKNN, accTestRunsKNN.length)[1];
		System.out.println("KNN Average testing accuracy: " + aveAccTestKNN);
		System.out.println("KNN Standard Deviation of testing accuracy: "
				+ stdAccTestKNN);


		System.out.println("");
		double aveTrainAccDT = NewMath.Mean_STD(accTrainRunsDT)[0];
		double stdTrainAccDT = NewMath.Mean_STD(accTrainRunsDT)[1];
		System.out.println("DT Average Training accuracy: " + aveTrainAccDT);
		System.out.println("DT Standard Deviation of Training accuracy:  "
				+ stdTrainAccDT);

		System.out.println("");
		double aveTrainAccNB = NewMath.Mean_STD(accTrainRunsNB)[0];
		double stdTrainAccNB = NewMath.Mean_STD(accTrainRunsNB)[1];
		System.out.println("NB Average Training accuracy: " + aveTrainAccNB);
		System.out.println("NB Standard Deviation of Training accuracy:  "
				+ stdTrainAccNB);

		System.out.println("");
		double aveTrainAccKNN = NewMath.Mean_STD(accTrainRunsKNN)[0];
		double stdTrainAccKNN = NewMath.Mean_STD(accTrainRunsKNN)[1];
		System.out.println("KNN Average Training accuracy: " + aveTrainAccKNN);
		System.out.println("KNN Standard Deviation of Training accuracy:  "
				+ stdTrainAccKNN);



		double CFOrg_aveAccTest_DT = NewMath.Mean_STD(CFOrgAccTestingRunsDT, CFOrgAccTestingRunsDT.length)[0];
		double CFOrg_stdAccTest_DT = NewMath.Mean_STD(CFOrgAccTestingRunsDT, CFOrgAccTestingRunsDT.length)[1];
		System.out.println("DT CFOrg_Average testing accuracy: "
				+ CFOrg_aveAccTest_DT);
		System.out.println("DT CFOrg_Standard Deviation of testing accuracy: "
				+ CFOrg_stdAccTest_DT);

		System.out.println("");
		double CFOrg_aveAccTest_NB = NewMath.Mean_STD(CFOrgAccTestingRunsNB, CFOrgAccTestingRunsNB.length)[0];
		double CFOrg_stdAccTest_NB = NewMath.Mean_STD(CFOrgAccTestingRunsNB, CFOrgAccTestingRunsNB.length)[1];
		System.out.println("NB CFOrg_Average testing accuracy: "
				+ CFOrg_aveAccTest_NB);
		System.out.println("NB CFOrg_Standard Deviation of testing accuracy: "
				+ CFOrg_stdAccTest_NB);

		System.out.println("");
		double CFOrg_aveAccTest_KNN = NewMath.Mean_STD(CFOrgAccTestingRunsKNN, CFOrgAccTestingRunsKNN.length)[0];
		double CFOrg_stdAccTest_KNN = NewMath.Mean_STD(CFOrgAccTestingRunsKNN, CFOrgAccTestingRunsKNN.length)[1];
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
		double CFOrg_aveTrainAcc_NB = NewMath.Mean_STD(CFOrgAccTrainingRunsNB)[0];
		double CFOrg_stdTrainAcc_NB = NewMath.Mean_STD(CFOrgAccTrainingRunsNB)[1];
		System.out.println("NB_CFOrg_Average Training accuracy: "
				+ CFOrg_aveTrainAcc_NB);
		System.out
				.println("NB_CFOrg_Standard Deviation of Training accuracy:  "
						+ CFOrg_stdTrainAcc_NB);

		System.out.println("");
		double CFOrg_aveTrainAcc_KNN = NewMath
				.Mean_STD(CFOrgAccTrainingRunsKNN)[0];
		double CFOrg_stdTrainAcc_KNN = NewMath
				.Mean_STD(CFOrgAccTrainingRunsKNN)[1];
		System.out.println("KNN_CFOrg_Average Training accuracy: "
				+ CFOrg_aveTrainAcc_KNN);
		System.out
				.println("KNN_CFOrg_Standard Deviation of Training accuracy:  "
						+ CFOrg_stdTrainAcc_KNN);



		this.printFile(dir + "/" + fname + "CFDTTesting.txt", accTestRunsDT);
		this.printFile(dir + "/" + fname + "CFOrgDTTesting.txt",CFOrgAccTestingRunsDT);

		this.printFile(dir + "/" + fname + "CFKNNTesting.txt", accTestRunsKNN);
		this.printFile(dir + "/" + fname + "CFOrgKNNTesting.txt", CFOrgAccTestingRunsKNN);

		this.printFile(dir + "/" + fname + "CFNBTesting.txt", accTestRunsNB);
		this.printFile(dir + "/" + fname + "CFOrgNBTesting.txt", CFOrgAccTestingRunsNB);



		this.printFile(dir + "/" + fname + "CFDTTraining.txt", accTrainRunsDT);
		this.printFile(dir + "/" + fname + "CFOrgDTTraining.txt",CFOrgAccTrainingRunsDT);

		this.printFile(dir + "/" + fname + "CFKNNTraining.txt", accTrainRunsKNN);
		this.printFile(dir + "/" + fname + "CFOrgKNNTraining.txt", CFOrgAccTrainingRunsKNN);

		this.printFile(dir + "/" + fname + "CFNBTraining.txt", accTrainRunsNB);
		this.printFile(dir + "/" + fname + "CFOrgNBTraining.txt", CFOrgAccTrainingRunsNB);

		this.printOperatorFile(dir + "/" + fname + "Operators.txt", operatorsRuns);

		this.printFile(dir + "/" + fname + "times.txt", timeRuns);

		double[] orgTest = OrgClassification.excuteClassification2(fname);

		double orgDT = orgTest[0];
		double orgKNN = orgTest[1];
		double orgNB = orgTest[2];

		double orgDTtr = orgTest[3];
		double orgKNNtr = orgTest[4];
		double orgNBtr = orgTest[5];
		System.out.println(orgDT + " " + orgKNN + " " + orgNB);

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

		System.out.println();
		System.out.println();
		System.out.println();

		LatexFormat.printBigLatexTable(fname, noFeatures, df, dg, orgDT,
				orgKNN, orgNB, accTestRunsDT, aveAccTestDT, stdAccTestDT,
				accTestRunsKNN, aveAccTestKNN, stdAccTestKNN, accTestRunsNB,
				aveAccTestNB, stdAccTestNB, CFOrgAccTestingRunsDT,
				CFOrg_aveAccTest_DT, CFOrg_stdAccTest_DT,
				CFOrgAccTestingRunsKNN, CFOrg_aveAccTest_KNN,
				CFOrg_stdAccTest_KNN, CFOrgAccTestingRunsNB,
				CFOrg_aveAccTest_NB, CFOrg_stdAccTest_NB);

	}



	private void printFile(String fname, double[] val) throws IOException{
		PrintWriter wf = new PrintWriter(new FileWriter(fname));
		for (int r = 0; r < number_of_runs; ++r) {
			wf.println(df.format(100 * val[r]));
		}
		wf.close();
	}

	private void printFile(String fname, long[] val) throws IOException{
		PrintWriter wf = new PrintWriter(new FileWriter(fname));
		for (int r = 0; r < number_of_runs; ++r) {
			wf.println(df.format(val[r]));
		}
		wf.close();
	}

	private void printOperatorFile(String fname, char[][][] val) throws IOException{
		PrintWriter wf = new PrintWriter(new FileWriter(fname));
		for (int r = 0; r < number_of_runs; ++r) {
			wf.println("run at: " + r);
			for(int j = 0; j < val[r].length; j++){
				for(int i = 0; i < val[r][j].length; i++){
					wf.print(val[r][j][i] + " ");
				}
				wf.println();
			}


		}
		wf.close();

	}



	private class ConstructedFeature{
		char[] operators;
		double[] features;
		public ConstructedFeature(char[] op, double[] ft){
			this.operators = op;
			this.features = ft;
		}
	}


	public static void main(String[] args){
		try {
			new MultiConst(args[0]);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
