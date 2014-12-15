package com.psofc.util;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.text.DecimalFormat;
import java.util.List;
import java.util.Random;


import com.psofc.HelpDataset;
import com.psofc.MyClassifier;
import com.psofc.util.examples.ArrayRepresentation;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

public class OutputYan {

	String fname = "";

	public OutputYan(String dir, String fname) {
		this.fname += dir + "/";
		this.fname += fname;
	}

	public void printOperators(char[][] operators, int number_of_runs) throws IOException {
		String fn = fname + "_OperatorsRuns.txt";
		PrintWriter wf = new PrintWriter(new FileWriter(fn));

		for (int r = 0; r < number_of_runs; ++r) {
			for (int j = 0; j < operators[0].length; ++j) {
				wf.print(operators[r][j]);
			}
			wf.println();
			wf.println("=================");
		}

		wf.close();

	}


	public void printConstructedFeature(double[][] tr, double[][] tt, int number_of_runs) throws IOException {
		String fn = fname + "_trCFRuns.txt";
		String fn2 = fname + "_ttCFRuns.txt";
		PrintWriter wf = new PrintWriter(new FileWriter(fn));
		PrintWriter wf2 = new PrintWriter(new FileWriter(fn2));


		for (int r = 0; r < number_of_runs; ++r) {
			for (int j = 0; j < tr[0].length; ++j) {
				wf.println(tr[r][j]);
			}
			wf.println();
			wf.println();

			for (int j = 0; j < tt[0].length; ++j) {
				wf2.println(tt[r][j]);
			}
			wf2.println();
			wf2.println();
		}

		wf.close();
		wf2.close();

	}

//	public void printYan(int number_of_iterations, int number_of_runs,
//			double[] averageGbestIterations, double[] aveErTrainIterations,
//			DecimalFormat df, DecimalFormat dg, double aveTrainAcc,
//			double aveAccTest, double stdAccTest, double stdTrainAcc,
//			double fullTest, double aveFitness, double stdFitness,
//			int bestRun, int dimension, double fullTrain, ArrayRepresentation.ArrRecord arrRecord) throws IOException {
//
//		/**
//		 * Output the results to txt
//		 */
//		String[] files = { fname + "_FitPlot.txt", fname + "_TrErPlot.txt",
//				fname + "_TtestTeAcc.txt", fname + "_Time.txt",
//				fname + "_aaBestPosFit.txt", fname + "CFNoCTraining.txt",
//				fname + "CFNoCTesting.txt", fname + "CForgNoCTraining.txt",
//				fname + "CFOrgNoCTesting.txt", fname + "CFDTTraining.txt",
//				fname + "CFDTTesting.txt", fname + "CForgDTTraining.txt",
//				fname + "CFOrgDTTesting.txt", fname + "CFNBTraining.txt",
//				fname + "CFNBTesting.txt", fname + "CForgNBTraining.txt",
//				fname + "CFOrgNBTesting.txt", fname + "CFKNNTraining.txt",
//				fname + "CFKNNTesting.txt", fname + "CForgKNNTraining.txt",
//				fname + "CFOrgKNNTesting.txt" }; // 4---7
//
//		for (int i = 0; i < files.length; i++) {
//			PrintWriter wf = new PrintWriter(new FileWriter(files[i]));
//			switch (i) {
//			case 0:
//				for (int j = 0; j < number_of_iterations; ++j) {
//					wf.println(averageGbestIterations[j]); // Change of the
//															// fitness;
//				}
//				break;
//
//			case 1:
//				for (int j = 0; j < number_of_iterations; ++j) {
//					wf.println(aveErTrainIterations[j]); // change of the
//															// (error) ;
//				}
//				break;
//
//			case 2:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * arrRecord.YanaccTestRuns[r]));
//				}
//				break;
//
//			case 3:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(arrRecord.timeRuns[r]);
//				}
//				break;
//
//			case 4:
//				wf.println(" &" + df.format(100 * aveAccTest) + " ("
//						+ df.format(100 * arrRecord.bestAccTest) + ")" + " &"
//						+ dg.format(stdAccTest) + " &"
//						+ df.format(100 * aveTrainAcc) + " &"
//						+ dg.format(stdTrainAcc));
//				wf.println("");
//				wf.println("fullSet.Testacc()   " + " & "
//						+ df.format(100 * fullTest));
//				wf.println("");
//				wf.println("-------------------------------------------------------------------------------------------------");
//				wf.println("Best-Acc      Ave-Acc      Std-Acc      Ave-Fitness      Std-Fitness");
//				wf.println(" &" + df.format(100 * bestAccTest) + " &"
//						+ df.format(100 * aveAccTest) + " &"
//						+ dg.format(stdAccTest) + " &" + dg.format(aveFitness)
//						+ " &" + dg.format(stdFitness));
//				wf.println("-------------------------------------------------------------------------------------------------");
//				wf.println(" Ave-Acc (Best-Acc)      Std-Acc      Ave-TrainAcc      Std-TrainAcc");
//				wf.println(" &" + df.format(100 * aveAccTest) + " ("
//						+ df.format(100 * bestAccTest) + ")" + " &"
//						+ dg.format(stdAccTest) + " &"
//						+ df.format(100 * aveTrainAcc) + " &"
//						+ dg.format(stdTrainAcc));
//				wf.println();
//
//				wf.println("");
//				wf.println("fullSet.Tainacc() " + fullTrain);
//				wf.println("fullSet.Testacc()   " + fullTest);
//
//				wf.println("-------------------------------------------------------------------------------------------------");
//				wf.println("Fitness              TrainAcc                    Acc_Test  (in each run)");
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(bestFitnessRuns[r] + "   " + YanaccTrainRuns[r]
//							+ "         " + YanaccTestRuns[r]);
//				}
//				wf.println("");
//				wf.println("Job for the Best testing acc is:  " + (bestRun + 1)
//						+ " Run (index+1) ");
//				wf.println("position for the Best testing accuracy is: ");
//				for (int d = 0; d < dimension; d++) {
//					wf.print(bestPositionRuns[bestRun][d] + "  ");
//				}
//				wf.println("");
//				wf.println("Best testing accuracy: " + bestAccTest);
//				wf.println("");
//				wf.println("Average testing accuracy: " + aveAccTest);
//				wf.println("Standard Deviation of testing accuracy: "
//						+ stdAccTest);
//				wf.println("");
//				wf.println("Average Training accuracy: " + aveTrainAcc);
//				wf.println("Standard Deviation of Training accuracy:  "
//						+ stdTrainAcc);
//				wf.println("");
//				wf.println("Average fitness (Error): " + aveFitness);
//				wf.println("Standard Deviation of fitness (Error): "
//						+ stdFitness);
//				wf.println("");
//				wf.println("-------------------------------------------------------------------------------------------------");
//				wf.println("Best-Acc      Ave-Size      Ave-Acc      Std-Acc      Ave-Fitness      Std-Fitness");
//				wf.println(" &" + df.format(100 * bestAccTest) + " &"
//						+ df.format(100 * aveAccTest) + " &"
//						+ dg.format(stdAccTest) + " &" + dg.format(aveFitness)
//						+ " &" + dg.format(stdFitness));
//				wf.println("-------------------------------------------------------------------------------------------------");
//				wf.println("Ave-Acc (Best-Acc)      Std-Acc      Ave-TrainAcc      Std-TrainAcc");
//				wf.println(" &" + df.format(100 * aveAccTest) + " ("
//						+ df.format(100 * bestAccTest) + ")" + " &"
//						+ dg.format(stdAccTest) + " &"
//						+ df.format(100 * aveTrainAcc) + " &"
//						+ dg.format(stdTrainAcc));
//				wf.println();
//				wf.println();
//
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println((r + 1) + "  Run");
//					for (int d = 0; d < dimension; ++d) {
//						wf.print(bestPositionRuns[r][d] + "  ");
//					}
//					wf.println();
//				}
//
//				wf.println();
//				wf.println();
//				wf.println("Best fitness in each Job");
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(bestFitnessRuns[r]);
//				}
//				wf.println();
//				wf.println();
//				wf.println("Train accuracy in each Job");
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(YanaccTrainRuns[r]);
//				}
//				wf.println();
//				wf.println();
//				wf.println("Test accuarcy in each Job");
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(YanaccTestRuns[r]);
//				}
//
//				wf.println();
//				wf.println();
//				wf.println("Gbest fitness in each Job in all generations:");
//				for (int r = 0; r < number_of_runs; ++r) {
//					for (int j = 0; j < number_of_iterations; ++j) {
//						wf.println(gbestRunsIterations[r][j]);
//					}
//					wf.println();
//					wf.println();
//				}
//
//				wf.println();
//				wf.println("Error rate of Gbest in each Job in all generations:");
//				for (int r = 0; r < number_of_runs; ++r) {
//					for (int j = 0; j < number_of_iterations; ++j) {
//						wf.println(df
//								.format(100 * eroGbestRunsIterations[r][j]));
//					}
//					wf.println();
//					wf.println();
//				}
//
//				break;
//
//			case 5:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * YanaccTrainRuns[r]));
//				}
//				break;
//
//			case 6:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * YanaccTestRuns[r]));
//				}
//				break;
//
//			case 7:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * YanCFOrgaccTrainRuns[r]));
//				}
//				break;
//
//			case 8:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * YanCFOrgaccTestRuns[r]));
//				}
//				break;
//
//			case 9:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * DTaccTrainRuns[r]));
//				}
//				break;
//
//			case 10:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * DTaccTestRuns[r]));
//				}
//				break;
//
//			case 11:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * DTCFOrgaccTrainRuns[r]));
//				}
//				break;
//
//			case 12:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * DTCFOrgaccTestRuns[r]));
//				}
//				break;
//
//			case 13:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * NBaccTrainRuns[r]));
//				}
//				break;
//
//			case 14:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * NBaccTestRuns[r]));
//				}
//				break;
//
//			case 15:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * NBCFOrgaccTrainRuns[r]));
//				}
//				break;
//
//			case 16:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * NBCFOrgaccTestRuns[r]));
//				}
//				break;
//
//			case 17:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * KNNaccTrainRuns[r]));
//				}
//				break;
//
//			case 18:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * KNNaccTestRuns[r]));
//				}
//				break;
//
//			case 19:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * KNNCFOrgaccTrainRuns[r]));
//				}
//				break;
//
//			case 20:
//				for (int r = 0; r < number_of_runs; ++r) {
//					wf.println(df.format(100 * KNNCFOrgaccTestRuns[r]));
//				}
//				break;
//
//			default:
//				throw new AssertionError();
//			}
//			wf.close();
//		}
//
//	}

	public Dataset constructNewDataset(Dataset data, double[] features,
			char[] operators) {

		Dataset fcdata = HelpDataset.removeFeatures(data, features);

		// for(int i = 0; i < operators.length;
		// i++){System.out.print(operators[i] + " ");}
		// System.out.println();
		// for(int rm = 0; rm < fcdata.size(); rm++){
		// Instance line = fcdata.get(rm);
		// for(int i = 0; i < line.size(); i++){
		// System.out.print(line.get(i) + ",");
		// }
		// // System.out.println(line.classValue());
		// System.out.println();
		// }
		// System.out.println("===================================================");
		fcdata = FCFunction.calConstructFeaBing(fcdata, operators);

		// for(int rm = 0; rm < fcdata.size(); rm++){
		// Instance line = fcdata.get(rm);
		// for(int i = 0; i < line.size(); i++){
		// System.out.print(line.get(i) + ",");
		// }
		// // System.out.println(line.classValue());
		// System.out.println();
		// }

		Dataset superdata = new DefaultDataset();

		List<Instance> instances = fcdata.subList(0, fcdata.size());
		List<Instance> ori = data.subList(0, data.size());

		double[] f = new double[features.length + 1];
		for (int rm = 0; rm < ori.size(); rm++) {
			Instance line = ori.get(rm);

			for (int x = 0; x < features.length; x++) {
				f[x] = line.value(x);
			}
			f[features.length] = instances.get(rm).value(0);

			Instance inst = new DenseInstance(f, line.classValue());
			superdata.add(inst);
			// wt += instances.get(rm).value(0) + "," +
			// instances.get(rm).classValue() + "\n";
		}

		return superdata;
	}

	public double[] superDataTestingDT(Dataset superdata) {
		Dataset[] folds = superdata.folds(10, new Random(100));

		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();

		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			testing.addAll(folds[te[i]]);
		}

		double accuracyTr = 0.0;
		double accuracyTt = 0.0;

		MyClassifier mc = new MyClassifier(new Random(1));

		mc.ClassifierDT();
		accuracyTt = mc.classify(training, testing);
		accuracyTr = mc.classify(training, training);

		double[] result = { accuracyTr, accuracyTt };
		return result;

	}

	public double[] superDataTestingKNN(Dataset superdata) {
		Dataset[] folds = superdata.folds(10, new Random(100));

		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();

		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			testing.addAll(folds[te[i]]);
		}

		double accuracyTr = 0.0;
		double accuracyTt = 0.0;

		MyClassifier mc = new MyClassifier(new Random(1));

		mc.ClassifierKNN();
		try {
			accuracyTt = mc.classify(training, testing);
		} catch (Exception e) {
			e.printStackTrace();
			accuracyTt = 0.0;
		}
		try {
			accuracyTr = mc.classify(training, training);
		} catch (Exception e) {
			e.printStackTrace();
			accuracyTr = 0.0;
		}

		double[] result = { accuracyTr, accuracyTt };
		return result;

	}

	public double[] superDataTestingNB(Dataset superdata) {
		Dataset[] folds = superdata.folds(10, new Random(100));

		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();

		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			testing.addAll(folds[te[i]]);
		}

		double accuracyTr = 0.0;
		double accuracyTt = 0.0;

		MyClassifier mc = new MyClassifier(new Random(1));

		mc.ClassifierNB();
		accuracyTt = mc.classify(training, testing);
		accuracyTr = mc.classify(training, training);

		double[] result = { accuracyTr, accuracyTt };
		return result;

	}

	public double[] superDataTestingYan(Dataset superdata) {
		Dataset[] folds = superdata.folds(10, new Random(100));

		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();

		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			testing.addAll(folds[te[i]]);
		}

		double accuracyTr = 0.0;
		double accuracyTt = 0.0;

		accuracyTt = ClassifierYan.classify(training, testing);
		accuracyTr = ClassifierYan.classify(training, training);

		double[] result = { accuracyTr, accuracyTt };
		return result;

	}

	/**
	 * testing the cf + org
	 *
	 * @param superdata
	 */
	public double[] superDataTestingAcc(Dataset superdata, String type) {

		// for(int rm = 0; rm < superdata.size(); rm++){
		// Instance line = superdata.get(rm);
		// for(int i = 0; i < 15; i++){
		// System.out.print(line.get(i) + ",");
		// }
		// System.out.println(line.classValue());
		// }
		double[] result = new double[3];
		Dataset[] folds = superdata.folds(10, new Random(100));

		Dataset training = new DefaultDataset();
		Dataset testing = new DefaultDataset();

		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			training.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			testing.addAll(folds[te[i]]);
		}

		double accuracy = 0.0;

		MyClassifier mc = new MyClassifier(new Random(1));

		mc.ClassifierDT();
		accuracy = mc.classify(training, testing);
		result[0] = (int) (accuracy * 10000) / 100.0;
		System.out
				.println(type
						+ " The best testing accuracy using both the original features and the constructed features are: ");
		System.out.println(type + " DT: " + ((int) (accuracy * 10000)) / 100.0);

		mc.ClassifierKNN();
		accuracy = mc.classify(training, testing);
		result[1] = (int) (accuracy * 10000) / 100.0;
		System.out
				.println(type + " 5NN: " + ((int) (accuracy * 10000)) / 100.0);

		mc.ClassifierNB();
		accuracy = mc.classify(training, testing);
		result[2] = (int) (accuracy * 10000) / 100.0;
		System.out.println(type + " NB: " + ((int) (accuracy * 10000)) / 100.0);

		// out.append(accuracy + "\n");
		//
		// out.flush();
		return result;

	}

}
