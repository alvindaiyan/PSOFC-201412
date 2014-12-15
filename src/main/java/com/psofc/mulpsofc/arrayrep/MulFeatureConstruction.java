package com.psofc.mulpsofc.arrayrep;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

import com.psofc.singlepsofc.arrayrep.OperatorSelection;


import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;
import net.sf.javaml.tools.data.FileHandler;

import com.psofc.HelpDataset;
import com.psofc.Problem;
import com.psofc.util.FCFunction;
import com.psofc.util.NewMath;
import com.psofc.util.ReadResults;

public class MulFeatureConstruction extends Problem {

	private Object clz;


    public Object getClz() {
		return clz;
	}




	public void setClz(Object clz) {
		this.clz = clz;
	}




	public MulFeatureConstruction() throws IOException {
        System.out.println("mulFeatureConstruction ");
        setMinimization(true);
        setMaxDomain(1.0);
        setMinDomain(0.0);
        setMaxVelocity(0.1);
        setMinVelocity(-0.1);
        setThreshold(0.5);
        System.out.println("setThreshold(0.5);");
        setBinaryContin(Boolean.FALSE);
    }




	@Override
	public double fitness(List<Double> position) {
		double[] features = NewMath.Position2FeaturesArray(position);

		int sum = FCFunction.noFeature(features);

		if(sum == 0){return 1000000;}



		char[] operators = OperatorSelection.getOpFromArray(position);


		Dataset dataTem = this.getTraining().copy();

		dataTem = HelpDataset.removeFeatures(dataTem, features);

		dataTem = FCFunction.calConstructFeaBing(dataTem, operators);


		Dataset[] folds = dataTem.folds((this.getNumFolds()), new Random(100));
		Dataset dataTrain = new DefaultDataset();
		Dataset dataTest = new DefaultDataset();

		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
//		int[] tr = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
//		int[] te = {4}; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			dataTrain.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			dataTest.addAll(folds[te[i]]);
		}


		Map<Object, Integer> count = new HashMap<Object, Integer>();


		Object[] clzz = new Object[dataTem.classes().size()];
    	int index = 0;
    	for(Object o : dataTem.classes()){
    		clzz[index] = o;
    		index++;
    		count.put(o, 0);

    	}

//    	Object cz = clzz[0]; // TODO: Test class

		Interval inv = this.findInterval(dataTrain, clz);

		int totalInstance = 0;

		for(int i = 0; i < dataTest.size(); i++){
			if(inv.in(dataTest.get(i).value(0))){
				totalInstance++;

				if(!count.containsKey(dataTest.get(i).classValue())){throw new IllegalArgumentException("no such label");}

				count.put(dataTest.get(i).classValue(), count.get(dataTest.get(i).classValue()) + 1);
			}
		}

		double[] p = new double[clzz.length];

		for(int i = 0; i < clzz.length; i++){
			p[i] = (count.get(clzz[i]) + 0.0) / (totalInstance + 0.0);
		}


		double fitness = 0;

		for(double v:p){
			if(v == 0){continue;}
//			System.err.println(p);
			fitness = fitness - v * Math.log(v);
		}

		return fitness;
	}


	public double testFitness(Dataset dataTem, Object cz){
		Dataset[] folds = dataTem.folds((this.getNumFolds()), new Random(100));
		Dataset dataTrain = new DefaultDataset();
		Dataset dataTest = new DefaultDataset();

		int[] tr = { 0, 2, 3, 5, 6, 8, 9 };
		int[] te = { 1, 4, 7 }; // 7, 4 and 6,5 changes
//		int[] tr = { 0, 1, 2, 3, 5, 6, 7, 8, 9 };
//		int[] te = {4}; // 7, 4 and 6,5 changes
		for (int i = 0; i < tr.length; i++) {
			dataTrain.addAll(folds[tr[i]]);
		}
		for (int i = 0; i < te.length; i++) {
			dataTest.addAll(folds[te[i]]);
		}


		Map<Object, Integer> count = new HashMap<Object, Integer>();


		Object[] clzz = new Object[dataTem.classes().size()];
    	int index = 0;
    	for(Object o : dataTem.classes()){
    		clzz[index] = o;
    		index++;
    		count.put(o, 0);

    	}

//    	Object cz = clzz[0]; // TODO: Test class

		Interval inv = this.findInterval(dataTrain, clz);

		int totalInstance = 0;

		for(int i = 0; i < dataTest.size(); i++){
			if(inv.in(dataTest.get(i).value(0))){
				totalInstance++;

				if(!count.containsKey(dataTest.get(i).classValue())){throw new IllegalArgumentException("no such label");}

				count.put(dataTest.get(i).classValue(), count.get(dataTest.get(i).classValue()) + 1);
			}
		}

		double[] p = new double[clzz.length];

		for(int i = 0; i < clzz.length; i++){
			p[i] = (count.get(clzz[i]) + 0.0) / (totalInstance + 0.0);
		}


		double fitness = 0;

		for(double v:p){
			if(v == 0){continue;}
//			System.err.println(v);
			fitness = fitness - v * Math.log(v);
		}

		return fitness;
	}





	public Interval findInterval(Dataset data, Object clazz){
		List<Double> left = new ArrayList<Double>();
		left.add(Double.POSITIVE_INFINITY);
		List<Double> right = new ArrayList<Double>();
		right.add(Double.NEGATIVE_INFINITY);

		for(int i = 0; i < data.size(); i++){
			if(data.get(i).classValue().equals(clazz)){
				if(data.get(i).value(0) < this.findMax(left)){
					if(left.size() >= data.classes().size() / 200){
						left.remove(this.findMax(left));
					}
					left.add(data.get(i).value(0));

				}

				if(data.get(i).value(0) > this.findMin(right)){
					if(right.size() >= this.findMin(right)){
						if(right.size() >= data.classes().size() / 200){
							right.remove(this.findMin(right));
						}
						right.add(data.get(i).value(0));
					}
				}
			}
		}


		return new Interval(this.findMax(left), this.findMin(right));
	}


	private double findMax(List<Double> list){
		double val = Double.NEGATIVE_INFINITY;
		for(double d:list){
			val = Math.max(val, d);
		}
		return val;
	}

	private double findMin(List<Double> list){
		double val = Double.POSITIVE_INFINITY;
		for(double d:list){
			val = Math.min(val, d);
		}
		return val;
	}


	public class Interval{
		double left;
		double right;

		public Interval(double left, double right){
			this.left = left;
			this.right = right;
		}

		public double getLeft() {
			return left;
		}
		public void setLeft(double left) {
			this.left = left;
		}
		public double getRight() {
			return right;
		}
		public void setRight(double right) {
			this.right = right;
		}

		public boolean in(double d){
			return d >= left && d <= right;
		}

		@Override
		public String toString(){
			return "[" + left + ", " + right +"]";
		}

	}




	/**
	 * Test Program: using the first feature as the constructed feature for testing
	 * the fitness function
	 * @param args
	 * @throws NumberFormatException
	 * @throws java.io.IOException
	 */
	public static void main(String[] args) throws NumberFormatException, IOException{
		MulFeatureConstruction mfc = new MulFeatureConstruction();
		String fname = "wbcd";
		int noFeatures = Integer.parseInt(ReadResults.read1Line("Data/" + fname
				+ "/noFeatures.txt"));
		Dataset data = FileHandler.loadDataset(new File("Data/" + fname
				+ "/Data.data"), noFeatures, ",");


		Object[] clzz = new Object[data.classes().size()];
    	int index = 0;
    	for(Object o : data.classes()){
    		clzz[index] = o;
    		index++;
    	}

    	System.out.println(clzz[1]);

		Dataset dt = new DefaultDataset();

		for(int i = 0; i < data.size(); i++){
			Instance ins = new DenseInstance(1);

			ins.put(0, data.get(i).get(1));

			ins.setClassValue(data.get(i).classValue());

			dt.add(ins);
		}


//		mfc.findInterval(dt, clzz[1]);

//		System.out.println("result: " + mfc.testFitness(dt));
	}



}
