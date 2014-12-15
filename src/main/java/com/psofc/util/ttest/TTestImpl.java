package com.psofc.util.ttest;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.Scanner;


public class TTestImpl {

	public static void main(String[] args) {

		String fn = "sonar";
		String fnameDT = "files/" + fn + "CFDTTesting.txt";
		String fnameKNN = "files/" + fn + "CFKNNTesting.txt";
		String fnameNB = "files/" + fn + "CFNBTesting.txt";

		String CFOrgfnameDT = "files/" + fn + "CFOrgDTTesting.txt";
		String CFOrgfnameKNN = "files/" + fn + "CFOrgKNNTesting.txt";
		String CFOrgfnameNB = "files/" + fn + "CFOrgNBTesting.txt";

		double fullTestAccDT = 71.43, fullTestAccKNN = 76.19, fullTestAccNB = 53.97;

		double fullTrainAcc;
		// String[] files = OrgClassification.listf("Data");
		//
		// if (files == null || files.length == 0) {
		// throw new IllegalArgumentException("No files");
		// }

		System.out.println(ttest(fnameDT, fullTestAccDT));

		System.out.println(ttest(CFOrgfnameDT, fullTestAccDT));

		System.out.println(ttest(fnameKNN, fullTestAccKNN));

		System.out.println(ttest(CFOrgfnameKNN, fullTestAccKNN));

		System.out.println(ttest(fnameNB, fullTestAccNB));

		System.out.println(ttest(CFOrgfnameNB, fullTestAccNB));

	}

	public static String ttest(String fn, double fullAccorg) {
		int numRuns = 50;
		double[] fullAcc = new double[numRuns];

		for (int i = 0; i < fullAcc.length; i++) {
			fullAcc[i] = fullAccorg;
		}

		double[] testAcc = new double[numRuns];

		Scanner sc;
		try {
			sc = new Scanner(new File(fn));
			int i = 0;
			while (sc.hasNext()) {
				testAcc[i++] = sc.nextDouble();
			}

			if (i != numRuns) {
				throw new IllegalArgumentException(
						"the array length is not right");
			}

		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}

		return Ttest.TtestBingNew(fullAcc, testAcc);
	}

	public static String test2(String fname, String dir, String classifier, String ds, double fullTestAccDT) {

		String fnameDT = dir + "/" + fname + ds+ classifier + "Testing.txt";
		return ttest(fnameDT, fullTestAccDT);

	}

	public static String testTr(String fname, String dir, String classifier, String ds, double fullTestAccDT) {

		String fnameDT = dir + "/" + fname + ds+ classifier + "Training.txt";
		return ttest(fnameDT, fullTestAccDT);

	}


	@Deprecated
	public static String ttest(double[] testAcc, double fullAccorg) {

		int numRuns = 50;
		double[] fullAcc = new double[numRuns];

		for (int i = 0; i < fullAcc.length; i++) {
			fullAcc[i] = fullAccorg;
		}

		if (testAcc.length != fullAcc.length) {
			throw new IllegalArgumentException("The array is not equals");
		}

		return Ttest.TtestBingNew(fullAcc, testAcc);
	}
}
