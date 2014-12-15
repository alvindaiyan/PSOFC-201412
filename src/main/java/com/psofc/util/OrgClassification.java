package com.psofc.util;

import java.io.File;
import java.io.IOException;
import java.text.DecimalFormat;
import java.util.Random;
import java.util.Scanner;

import com.psofc.MyClassifier;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.tools.data.FileHandler;

public class OrgClassification {


	public static void main2(String[] args) throws NumberFormatException, IOException{

		String[] files = OrgClassification.getAllDataName();

		if(files == null || files.length == 0){
			throw new IllegalArgumentException("No files");
		}

		for(String fname:files){
			excuteClassification(fname);
		}

	}

	public static void main(String[] args) throws IOException{

		String[] files = OrgClassification.getAllDataName();

		if(files == null || files.length == 0){
			throw new IllegalArgumentException("No files");
		}

		String out = "";
		Scanner sc = null;
		for(String fname:files){
			sc = new Scanner(new File("Data/"+ fname + "/noFeatures.txt"));
			String noF = sc.next();
			out += fname + ", " + noF + "\n";
		}

		System.out.println(out);

	}


	public static String[] getAllDataName(){
		return listf("Data");
	}


	public static String[] listf(String directoryName){

		File directory = new File(directoryName);


		File[] fList = directory.listFiles();
		String[] fname = new String[fList.length];
		int i = 0;
		for(File file : fList ){
			fname[i++] = file.getName();
		}

		return fname;
	}

	public static void excuteClassification(String fname) throws IOException{
//		String fname = "arrhythmia";
      int noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname + "/noFeatures.txt"));
      Dataset data = FileHandler.loadDataset(new File("Data/" + fname + "/Data.data"), noFeatures, ",");


      Dataset[] folds = data.folds((10), new Random(100));
      Dataset training = new DefaultDataset();
      Dataset testing = new DefaultDataset();
      int[] tr = {0, 2, 3, 5, 6, 8, 9};
      int[] te = {1, 4, 7};   // 7, 4 and 6,5 changes
      for (int i = 0; i < tr.length; i++) {
          training.addAll(folds[tr[i]]);
      }
      for (int i = 0; i < te.length; i++) {
          testing.addAll(folds[te[i]]);
      }


      MyClassifier mc = new MyClassifier(new Random(1));

      mc.ClassifierDT();

      double fullTrainDT = mc.fullclassify(training, training);
      double fullTestDT = mc.fullclassify(training, testing);
      System.out.println("fullSetDT.Trainacc() " + fullTrainDT);
      System.out.println("fullSetDT.Testacc()   " + fullTestDT);

      mc.ClassifierKNN();

      double fullTrainKNN = mc.fullclassify(training, training);
      double fullTestKNN = mc.fullclassify(training, testing);
      System.out.println("fullSetKNN.Trainacc() " + fullTrainKNN);
      System.out.println("fullSetKNN.Testacc()   " + fullTestKNN);


      mc.ClassifierNB();
      double fullTrainNB;
      double fullTestNB ;
      try{
      fullTrainNB = mc.fullclassify(training, training);
      fullTestNB = mc.fullclassify(training, testing);
      System.out.println("fullSetNB.Trainacc() " + fullTrainNB);
      System.out.println("fullSetNB.Testacc()   " + fullTestNB);
      }catch(Error e){fullTrainNB = 0; fullTestNB = 0;}

      DecimalFormat df = new DecimalFormat("##.##");
      System.out.println();

      System.out.print("\\multirow{3}{*}{" + fname + "}" + " & " + "\\multirow{3}{*}{" + noFeatures + "}" + " &  Org"  + " & " + df.format(fullTestDT * 100));

      System.out.print(" & " + " & " + " & "+ df.format(fullTestKNN * 100)  + " & "  + " & "  + " & " + df.format(fullTestNB * 100) + " & "  + " & "); // all best

      System.out.println(" \\" + "\\" +  "\\cline{3-12}");

      System.out.println();
	}


	public static double[] excuteClassification2(String fname) throws IOException{
//		String fname = "arrhythmia";
      int noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname + "/noFeatures.txt"));
      Dataset data = FileHandler.loadDataset(new File("Data/" + fname + "/Data.data"), noFeatures, ",");


      Dataset[] folds = data.folds((10), new Random(100));
      Dataset training = new DefaultDataset();
      Dataset testing = new DefaultDataset();
      int[] tr = {0, 2, 3, 5, 6, 8, 9};
      int[] te = {1, 4, 7};   // 7, 4 and 6,5 changes
      for (int i = 0; i < tr.length; i++) {
          training.addAll(folds[tr[i]]);
      }
      for (int i = 0; i < te.length; i++) {
          testing.addAll(folds[te[i]]);
      }


      MyClassifier mc = new MyClassifier(new Random(1));

      mc.ClassifierDT();

      double fullTrainDT = mc.fullclassify(training, training);
      double fullTestDT = mc.fullclassify(training, testing);
      System.out.println("fullSetDT.Trainacc() " + fullTrainDT);
      System.out.println("fullSetDT.Testacc()   " + fullTestDT);

      mc.ClassifierKNN();

      double fullTrainKNN = mc.fullclassify(training, training);
      double fullTestKNN = mc.fullclassify(training, testing);
      System.out.println("fullSetKNN.Trainacc() " + fullTrainKNN);
      System.out.println("fullSetKNN.Testacc()   " + fullTestKNN);


      mc.ClassifierNB();
      double fullTrainNB;
      double fullTestNB ;
      try{
      fullTrainNB = mc.fullclassify(training, training);
      fullTestNB = mc.fullclassify(training, testing);
      System.out.println("fullSetNB.Trainacc() " + fullTrainNB);
      System.out.println("fullSetNB.Testacc()   " + fullTestNB);
      }catch(Error e){fullTrainNB = 0; fullTestNB = 0;}

      DecimalFormat df = new DecimalFormat("##.##");
      System.out.println();

      System.out.print("\\multirow{3}{*}{" + fname + "}" + " & " + "\\multirow{3}{*}{" + noFeatures + "}" + " &  Org"  + " & " + df.format(fullTestDT * 100));

      System.out.print(" & " + " & " + " & "+ df.format(fullTestKNN * 100)  + " & "  + " & "  + " & " + df.format(fullTestNB * 100) + " & "  + " & "); // all best

      System.out.println(" \\" + "\\" +  "\\cline{3-12}");

      System.out.println();

      double[] result = {fullTestDT * 100, fullTestKNN * 100, fullTestNB * 100, fullTrainDT * 100, fullTrainKNN * 100, fullTrainNB * 100};
      return result;
	}


}
