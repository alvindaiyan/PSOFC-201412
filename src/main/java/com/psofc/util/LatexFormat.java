package com.psofc.util;

import com.psofc.util.ttest.TTestImpl;

import java.text.DecimalFormat;


public class LatexFormat {

	public static String dir = "";

	public static String dt, dtorg, knn, knnorg, nb, nborg;


	public static void printClassifier(String cf, String fname, int noFeatures,
			DecimalFormat df, DecimalFormat dg, double[] accTestRuns,
			double aveAccTest, double stdAccTest, double[] accTrainRuns,
			double aveTrainAcc, double stdTrainAcc,
			double[] CFOrgAccTestingRuns, double CFOrg_aveAccTest_,
			double CFOrg_stdAccTest_, double[] CFOrgAccTrainingRuns,
			double CFOrg_aveTrainAcc_, double CFOrg_stdTrainAcc_, double accorg, double accorgTr) {

		String ttestCFtt = TTestImpl.test2(fname, dir, cf, "CF",
				Double.parseDouble(df.format(accorg)));
		String ttestCFOrgtt = TTestImpl.test2(fname, dir, cf, "CFOrg",
				Double.parseDouble(df.format(accorg)));

//		String ttestCFtt = "";
//		String ttestCFOrgtt = "";

		String ttestCFtr = TTestImpl.testTr(fname, dir, cf, "CF",
				Double.parseDouble(df.format(accorgTr)));
		String ttestCFOrgtr = TTestImpl.testTr(fname, dir, cf, "CFOrg",
				Double.parseDouble(df.format(accorgTr)));

		System.out.print("\\multirow{3}{*}{" + fname + "}" + " & "
				+ "\\multirow{3}{*}{" + noFeatures + "}" + "&"
				+ "\\multirow{3}{*}{" + cf + "}" + " & Org" + " &"
				+ df.format(accorg) + "&");

		System.out.print(" & " + " & " + " & " + df.format(accorgTr) + " & " + " & " + " &"); // all
																		// best

		System.out.println(" \\" + "\\" + "\\cline{4-12}");

		System.out.print("& & & CF &" + df.format(aveAccTest * 100) + " & "
				+ df.format(NewMath.getBest(accTestRuns) * 100) + " & "
				+ dg.format(stdAccTest) + " & " + ttestCFtt);// this is the CF
																// best

		System.out.print(" & " + df.format(aveTrainAcc * 100)
				+ " & " + df.format(NewMath.getBest(accTrainRuns) * 100) + " & "
				+ dg.format(stdTrainAcc) + " & " + ttestCFtr);

		System.out.println(" \\" + "\\" + "\\cline{4-12}");

		System.out.print("& & & CFOrg &" + df.format(CFOrg_aveAccTest_ * 100)
				+ " & " + df.format(NewMath.getBest(CFOrgAccTestingRuns) * 100)
				+ " & " + dg.format(CFOrg_stdAccTest_) + " & " + ttestCFOrgtt);// this
		// is
		// the
		// CFOrg
		// best
		System.out.print(" & "
				+ df.format(NewMath.getBest(CFOrgAccTrainingRuns) * 100)
				+ " & " + df.format(CFOrg_aveTrainAcc_ * 100) + " & "
				+ dg.format(CFOrg_stdTrainAcc_) + " & " + ttestCFOrgtr);// this
																		// is
																		// the
		// CFOrg
		// best

		System.out.println(" \\" + "\\" + "\\cline{1-12}");

		setTtest(ttestCFtt, ttestCFOrgtt, cf);

	}

	private static void setTtest(String cf, String cforg, String classifier){
		if(classifier.equalsIgnoreCase("dt")){
			dt = cf;
			dtorg = cforg;
		}else if(classifier.equalsIgnoreCase("knn")){
			knn = cf;
			knnorg = cforg;
		} else if(classifier.equalsIgnoreCase("nb")){
			nb = cf;
			nborg = cforg;
		}else throw new IllegalArgumentException("no such thing");
	}

	public static String printBigLatexTable(String fname, int noFeatures, DecimalFormat df, DecimalFormat dg,
			double orgDT, double orgKNN, double orgNB,
			double[] accTestRunsDT, double aveAccTestDT, double stdAccTestDT,
			double[] accTestRunsKNN, double aveAccTestKNN, double stdAccTestKNN,
			double[] accTestRunsNB, double aveAccTestNB, double stdAccTestNB,
			double[] CFOrg_accTestRunsDT, double CFOrg_aveAccTestDT, double CFOrg_stdAccTestDT,
			double[] CFOrg_accTestRunsKNN, double CFOrg_aveAccTestKNN, double CFOrg_stdAccTestKNN,
			double[] CFOrg_accTestRunsNB, double CFOrg_aveAccTestNB, double CFOrg_stdAccTestNB) {

		String latex = "";

		System.out.print("\\multirow{3}{*}{" + fname + "}" + " & "
				+ "\\multirow{3}{*}{" + noFeatures + "}" + " & Org"
				 + " & " + df.format(orgDT));


		System.out.print(" & " + " & " + " & " + " & " + df.format(orgKNN) + " & "
				+ " & " + " & " + " & " + df.format(orgNB) + " & " + " & " + " & "); // all best


		System.out.println(" \\" + "\\" + "\\cline{3-15}");


		System.out.print("& & CF & "
				+ df.format(NewMath.getBest(accTestRunsDT) * 100) + " & "
				+ df.format(aveAccTestDT * 100) + " & "
				+ dg.format(stdAccTestDT) + " & " + dt);// this is the CF best
		System.out.print(" & "
				+ df.format(NewMath.getBest(accTestRunsKNN) * 100) + " & "
				+ df.format(aveAccTestKNN * 100) + " & "
				+ dg.format(stdAccTestKNN)  + " & " + knn);
		System.out.print(" & "
				+ df.format(NewMath.getBest(accTestRunsNB) * 100) + " & "
				+ df.format(aveAccTestNB * 100) + " & "
				+ dg.format(stdAccTestNB) + " & " + nb);

		System.out.println(" \\" + "\\" + "\\cline{3-15}");

		System.out.print("& & CFOrg & "
				+ df.format(NewMath.getBest(CFOrg_accTestRunsDT) * 100)
				+ " & " + df.format(CFOrg_aveAccTestDT * 100) + " & "
				+ dg.format(CFOrg_stdAccTestDT) + " & " + dtorg);// this is the CFOrg best
		System.out.print(" & "
				+ df.format(NewMath.getBest(CFOrg_accTestRunsKNN) * 100)
				+ " & " + df.format(CFOrg_aveAccTestKNN * 100) + " & "
				+ dg.format(CFOrg_stdAccTestKNN) + " & " + knnorg);// this is the CFOrg best
		System.out.print(" & "
				+ df.format(NewMath.getBest(CFOrg_accTestRunsNB) * 100)
				+ " & " + df.format(CFOrg_aveAccTestNB * 100) + " & "
				+ dg.format(CFOrg_stdAccTestNB) + " & " + nborg);// this is the CFOrg best

		System.out.println(" \\" + "\\" + "\\cline{1-15}");

		System.out.println();

		return latex;
	}
}
