package com.psofc.mulpsofc.arrayrep;

import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;


import com.psofc.singlepsofc.arrayrep.OperatorSelection;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.tools.data.FileHandler;

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

public class MultiCont_AveFC {
	int number_of_runs = 50;
	double w = 0.729844;
	double c1 = 1.49618, c2 = 1.49618;
	int number_of_particles = 30;
	int number_of_iterations = 70;	// TODO: itrs
	Topology topology = new TopologyRing(30);
	int numFolds = 10;

	DecimalFormat df = new DecimalFormat("##.##");
	DecimalFormat dg = new DecimalFormat("##.#E0");


	public MultiCont_AveFC(String fn) throws NumberFormatException, IOException{
		String fname = fn;
		String dir = "file_multi_array";

		int noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname
				+ "/noFeatures.txt"));
		Dataset data = FileHandler.loadDataset(new File("Data/" + fname
				+ "/Data.data"), noFeatures, ",");
		int dimension = noFeatures * 2 - 1;

		MulFeatureConstruction problem = new MulFeatureConstruction();



		/** Sava the results */
		double[][][] gbestRunsIterations = new double[number_of_runs][data.classes().size()][number_of_iterations]; // get best fitness in each iterate in each run
		double[][][] eroGbestRunsIterations = new double[number_of_runs][data.classes().size()][number_of_iterations];

		double[][] bestFitnessRuns = new double[number_of_runs][data.classes().size()]; // get final bestfitnes in each run

		double[][][] bestPositionRuns = new double[number_of_runs][data.classes().size()][dimension];// get the position of best results in each run;



		char[][][][] operatorsRuns = new char[number_of_runs][data.classes().size()][dimension - 1][numFolds]; // temperory, this need to
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




		double[][] accTestRunsDT_run = new double[number_of_runs][numFolds]; // the best testing accuracy in each run
		double[][] accTrainRunsDT_run = new double[number_of_runs][numFolds];


		double[][] accTestRunsKNN_run = new double[number_of_runs][numFolds]; // the best testing accuracy in each run
		double[][] accTrainRunsKNN_run = new double[number_of_runs][numFolds];


		double[][] accTestRunsNB_run = new double[number_of_runs][numFolds]; // the best testing accuracy in each run
		double[][] accTrainRunsNB_run = new double[number_of_runs][numFolds];




		double[][] CFOrgAccTestingRunsDT_run = new double[number_of_runs][numFolds];
		double[][] CFOrgAccTrainingRunsDT_run = new double[number_of_runs][numFolds];

		double[][] CFOrgAccTestingRunsKNN_run = new double[number_of_runs][numFolds];
		double[][] CFOrgAccTrainingRunsKNN_run = new double[number_of_runs][numFolds];

		double[][] CFOrgAccTestingRunsNB_run = new double[number_of_runs][numFolds];
		double[][] CFOrgAccTrainingRunsNB_run = new double[number_of_runs][numFolds];


		long[][] timeRuns = new long[number_of_runs][numFolds];



		/**
		 * set up the training and testing data
		 */
		System.out.println("multi feature construction for average psofc" + fname + " with class number: " + data.classes().size());

		Dataset[] folds = data.folds(numFolds, new Random(100));
		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();

		/** Produce the random seeder */
		long[] Seeder = new long[number_of_runs];
		for (int r = 0; r < number_of_runs; r++) {
			Seeder[r] = r * r * r * 135 + r * r * 246 + 78;
		}


		/** start PSO */

		for (int r = 0; r < number_of_runs; ++r) {
			for(int numf = 0; numf < numFolds; numf++){
				training.clear();
				testing.clear();
				for(int i =  0; i < numFolds; i++){
					if(i != numf) training.addAll(folds[i]);
				}
				testing.addAll(folds[numf]);
				System.out.println("********************************** folds: "+ numf + "************************************************");
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
		//			s.getProblem().setFoldsTrain(foldsTrain);
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
						double bestFitness = s.getParticle(0).getNeighborhoodFitness();

						System.out.println(bestFitness);

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
							// System.arraycopy(s.getParticle(bestParticle).getNeighborhoodPosition().toArray(),
							// 0, bestPositionRuns[r], 0, dimension);
							for (int d = 0; d < s.getParticle(bestParticle)
									.getNeighborhoodPosition().size(); d++) {
								bestPositionRuns[r][constructedfeatures.size()][d] = s.getParticle(bestParticle)
										.getNeighborhoodPosition(d);
							}
						}

					} // end all iterations


					operatorsRuns[r][constructedfeatures.size()][numf] = OperatorSelection.getOpFromArray(bestPositionRuns[r][constructedfeatures.size()]);

					double[] features = NewMath.Position2FeaturesArray(bestPositionRuns[r][constructedfeatures.size()]);

					constructedfeatures.add(new ConstructedFeature(operatorsRuns[r][constructedfeatures.size()][numf], features));

				} // end of all classes

				long estimatedTime = System.currentTimeMillis() - initTime;

				timeRuns[r][numf] = estimatedTime;

				System.out.println("estimatedTime: " + estimatedTime);

				List<Dataset> dtrs = new ArrayList<Dataset>();
				List<Dataset> dtts = new ArrayList<Dataset>();


				int inx = 0;
				for(ConstructedFeature cf:constructedfeatures){
					System.err.println("feature " + inx++ + ":" );
					Dataset temTrain = training.copy();
					Dataset temTest = testing.copy();

					temTrain = HelpDataset.removeFeatures(temTrain, cf.features);
					temTrain = FCFunction.calConstructFeaBing(temTrain, cf.operators);

					dtrs.add(temTrain);


					for(Object tcz:data.classes()){
						System.err.println(problem.findInterval(temTrain, tcz));
						System.err.println(problem.testFitness(temTrain, tcz));
					}

					temTest = HelpDataset.removeFeatures(temTest, cf.features);
					temTest = FCFunction.calConstructFeaBing(temTest, cf.operators);

					dtts.add(temTest);
				}


				Dataset multTrain = FCFunction.constrMulDataset(dtrs);
				Dataset multTest = FCFunction.constrMulDataset(dtts);

				this.printCFs(dir + "/" + fname + "CFtr.txt", multTrain);
				this.printCFs(dir + "/" + fname + "CFtt.txt", multTest);



				int dataSetSize = noFeatures + data.classes().size();


				Dataset cforgTr = FCFunction.constrMulOrgDataset(training.copy(), multTrain, dataSetSize, noFeatures);
				Dataset cforgTt = FCFunction.constrMulOrgDataset(testing.copy(), multTest, dataSetSize, noFeatures);
				this.printCFs(dir + "/" + fname + "CFOrgtr.txt", cforgTr);
				this.printCFs(dir + "/" + fname + "CFOrgtt.txt", cforgTt);

				MyClassifier mc = new MyClassifier(new Random(1));
				mc.ClassifierDT();

				accTestRunsDT_run[r][numf] = mc.classify(multTrain, multTest);
				accTrainRunsDT_run[r][numf] = mc.classify(multTrain, multTrain);

				CFOrgAccTestingRunsDT_run[r][numf] = mc.classify(cforgTr, cforgTt);
				CFOrgAccTrainingRunsDT_run[r][numf] = mc.classify(cforgTr, cforgTr);


				mc.ClassifierKNN();
				accTestRunsKNN_run[r][numf] = mc.classify(multTrain,  multTest);
				accTrainRunsKNN_run[r][numf] = mc.classify(multTrain, multTrain);

				CFOrgAccTestingRunsKNN_run[r][numf] = mc.classify(cforgTr, cforgTt);
				CFOrgAccTrainingRunsKNN_run[r][numf] = mc.classify(cforgTr, cforgTr);

				mc.ClassifierNB();
				accTestRunsNB_run[r][numf] = mc.classify(multTrain, multTest);
				accTrainRunsNB_run[r][numf] = mc.classify(multTrain, multTrain);

				CFOrgAccTestingRunsNB_run[r][numf] = mc.classify(cforgTr, cforgTt);
				CFOrgAccTrainingRunsNB_run[r][numf] = mc.classify(cforgTr, cforgTr);



			} // end of all folds

			System.out.println("<<<<<<<<<=======================================================>>>>>>>>>>");

			accTestRunsDT[r] = NewMath.mean(accTestRunsDT_run[r]);
			accTrainRunsDT[r] = NewMath.mean(accTrainRunsDT_run[r]);

			CFOrgAccTestingRunsDT[r] = NewMath.mean(CFOrgAccTestingRunsDT_run[r]);
			CFOrgAccTrainingRunsDT[r] = NewMath.mean(CFOrgAccTrainingRunsDT_run[r]);

			accTestRunsKNN[r] = NewMath.mean(accTestRunsKNN_run[r]);
			accTrainRunsKNN[r] = NewMath.mean(accTrainRunsKNN_run[r]);

			CFOrgAccTestingRunsKNN[r] = NewMath.mean(CFOrgAccTestingRunsKNN_run[r]);
			CFOrgAccTrainingRunsKNN[r] = NewMath.mean(CFOrgAccTrainingRunsKNN_run[r]);


			accTestRunsNB[r] = NewMath.mean(accTestRunsNB_run[r]);
			accTrainRunsNB[r] = NewMath.mean(accTrainRunsNB_run[r]);

			CFOrgAccTestingRunsNB[r] = NewMath.mean(CFOrgAccTestingRunsNB_run[r]);
			CFOrgAccTrainingRunsNB[r] = NewMath.mean(CFOrgAccTrainingRunsNB_run[r]);

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


		this.printFile(dir + "/" + fname + "times.txt", timeRuns);

		this.printOperatorFile(dir + "/" + fname + "operators.txt", operatorsRuns);

		double[] orgTest = OrgClassification.excuteClassification2(fname);

		double orgDT = orgTest[0];
		double orgKNN = orgTest[1];
		double orgNB = orgTest[2];
		System.out.println(orgDT + " " + orgKNN + " " + orgNB);


		double orgDTtr = orgTest[3];
		double orgKNNtr = orgTest[4];
		double orgNBtr = orgTest[5];

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

	private void printFile(String fname, long[][] val) throws IOException{
		PrintWriter wf = new PrintWriter(new FileWriter(fname));
		for (int r = 0; r < number_of_runs; ++r) {

			for(int f = 0; f < numFolds; f++){
				wf.println(df.format(val[r][f]));
			}
			wf.println();
			wf.println();


		}
		wf.close();
	}

	private void printOperatorFile(String fname, char[][][][] val) throws IOException{
		PrintWriter wf = new PrintWriter(new FileWriter(fname));
		for (int r = 0; r < number_of_runs; ++r) {
			wf.println("run at: " + r);
			for(int j = 0; j < val[r].length; j++){
				wf.println("class at: " + j);
				for(int i = 0; i < val[r][j].length; i++){
					wf.println("folders at: " + i);
					for(int f = 0; f < val[r][j][i].length; f++){
						wf.print(val[r][j][i][f] + " ");
					}
					wf.println();
				}
				wf.println();
			}


		}
		wf.close();

	}

	static Map<String, Boolean> firstRun = new HashMap<String, Boolean>();;

	private void printCFs(String fname, Dataset dt) throws IOException{


		PrintWriter wf = new PrintWriter(new BufferedWriter(new FileWriter(fname, true)));
		if(!firstRun.containsKey(fname)){
			PrintWriter writer = new PrintWriter(new FileWriter(fname));
			writer.print("");
			writer.close();
			firstRun.put(fname, false);
		}

		wf.append(dt.toString());
		wf.append("\n");
		wf.close();
	}


	public class ConstructedFeature{
		char[] operators;
		double[] features;
		public ConstructedFeature(char[] op, double[] ft){
			this.operators = op;
			this.features = ft;
		}
	}

	public static void main(String[] args){
		try {
			new MultiCont_AveFC(args[0]);
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
