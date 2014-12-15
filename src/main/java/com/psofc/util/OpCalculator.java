package com.psofc.util;

import java.io.File;
import java.io.FileNotFoundException;
import java.text.DecimalFormat;
import java.util.Scanner;

public class OpCalculator {

	public static void main(String[] args) {

		String[] files = OrgClassification.getAllDataName();
		String dir = "file_pair/";
		DecimalFormat df = new DecimalFormat("##.##");
		System.out.println("Fname, avgPlus, avgMinus, avgMul, avgDiv");
		for (String file : files) {
			double[][] val = null;
			try {
				val = new OpCalculator().getOpPercentage(dir, file);
			} catch (FileNotFoundException e) {
				System.err.println("File " + file + " does not exist.");
			}

			if(val != null){
				System.out.print(file + ", ");
				String avgPlus = df.format(NewMath.mean(val[0]));
				String avgMinus = df.format(NewMath.mean(val[1]));
				String avgMulti = df.format(NewMath.mean(val[2]));
				String avgDiv = df.format(NewMath.mean(val[3]));
				System.out.println(avgPlus + ", " + avgMinus + ", " + avgMulti + ", " + avgDiv);
			}
		}
	}

	public double[][] getOpPercentage(String dir, String fname) throws FileNotFoundException {
		String f = dir + fname + "_OperatorsRuns.txt";
		Scanner sc = new Scanner(new File(f));
		double[][] val = new double[4][50];


		int i = 0;
		while (sc.hasNext()) {
			String line = sc.nextLine();
			char[] ops = line.toCharArray();
			double[] result = countOp(ops);
			int total = 0;
			for (double r : result) {
				total += r;
				// System.out.print(i + " ");
			}
//			val[i][0] = df.format(100.00 * result[0] / total);
//			val[i][1] = df.format(100.00 * result[1] / total);
//			val[i][2] = df.format(100.00 * result[2] / total);
//			val[i][3] = df.format(100.00 * result[3] / total);


			val[0][i] = 100.00 * result[0] / total;
			val[1][i] = 100.00 * result[1] / total;
			val[2][i] = 100.00 * result[2] / total;
			val[3][i] = 100.00 * result[3] / total;
			sc.nextLine();
			i++;
		}
//		System.out.println("index ends at " + i);
		return val;
	}

	public double[] countOp(char[] ops) {
		double[] val = new double[4];
		for (int i = 0; i < val.length; i++) {
			val[i] = 0.0;
		}
		for (char op : ops) {
			switch (op) {
			case '+':
				val[0]++;
				break;
			case '-':
				val[1]++;
				break;
			case '*':
				val[2]++;
				break;
			case '/':
				val[3]++;
				break;
			case 'a':
				return val;
			}

		}

		return val;
	}

}
