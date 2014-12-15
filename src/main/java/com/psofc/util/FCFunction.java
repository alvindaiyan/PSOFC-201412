/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc.util;

import java.util.List;

import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

/**
 *
 * @author xuebing
 */
public class FCFunction {


	/*
	 * construct new multiple feature by a set of new datasets
	 */
	public static Dataset constrMulDataset(List<Dataset> data){

		Dataset newData = new DefaultDataset();
		for(int i = 0; i < data.get(0).size(); i++){

			Instance ins = new DenseInstance(data.size());

			for(int j = 0; j < data.size(); j++){
				ins.put(j, data.get(j).get(i).get(0));
			}

			ins.setClassValue(data.get(0).get(i).classValue());

			newData.add(ins);
		}

		return newData;
	}


	public static Dataset constrMulOrgDataset(Dataset org, Dataset mul, int dataSetSize, int noFeatures){
		// multi + org datast
		Dataset cforgTr = new DefaultDataset();
		for (int i = 0; i < mul.size(); i++) {
			Dataset temTrain = org.copy();

			Instance ins = new DenseInstance(dataSetSize);
			int idx = 0;
			for (int j = 0; j < noFeatures; j++) {
				ins.put(idx++,temTrain.get(i).get(j));
			}

			for (int j = 0; j < org.classes().size(); j++) {
				ins.put(idx++,mul.get(i).get(j));
			}

			if (!temTrain.get(i).classValue().equals(mul.get(i).classValue())) {
				System.err.println("This is not equal instance: "
						+ temTrain.get(i).classValue() + " != "
						+ mul.get(i).classValue());
			}
			ins.setClassValue(temTrain.get(i).classValue());
			cforgTr.add(ins);
		}

		return cforgTr;
	}


    /*change the datata according the constructed features
     */
    public static Dataset calConstructFeaBing(Dataset data, char[] operators) {

        int noAttri = data.noAttributes();


        Dataset newData = new DefaultDataset();

        for (int da = 0; da < data.size(); da++) {

            Instance newInst = new DenseInstance(1);  // only include one constructed feature

            DenseInstance instanc = (DenseInstance) data.get(da);

            double newValue = instanc.get(0);

            for (Integer j = 1; j < noAttri; j++) { // calculate from the second value

                newValue = calculate(newValue, instanc.get(j), operators[j - 1]);
//                System.out.print(" "+newValue);
            }
//            System.out.println();
//            newValue = newValue / 10E155;
//            if(newValue == 0.0){newValue = -0.07106426385583135;}
            newInst.put(0, newValue);
            newInst.setClassValue(instanc.classValue());

            newData.add(newInst);
        }

        newData = dataNomarlise.DataNormalisation(newData);

        return newData;
    }

    /*calculate the new value interface the construction function
     */
    public static double calculate(double calEd, double toCal, char operator) {

        double newValue = 0.0;

        if (operator == '+') {
            newValue = calEd + toCal;

        } else if (operator == '-') {
            newValue = calEd - toCal;

        } else if (operator == '*') {
            newValue = calEd * toCal;

        } else if (operator == '/') {
        	if(toCal==0){toCal = 1;}
            newValue = calEd / toCal;  // need to rewrite as a protected dividision
        }
        return newValue;
    }


    /*Simple sum all the values of each selected features
     */
    public static Dataset sum(Dataset data) {

        int noAttri = data.noAttributes();


        Dataset newData = new DefaultDataset();

        for (int da = 0; da < data.size(); da++) {

            Instance newInst = new DenseInstance(1);  // only include one constructed feature

            DenseInstance instanc = (DenseInstance) data.get(da);

            double sum = 0.0;
            for (Integer j = 0; j < noAttri; j++) {
                sum += instanc.get(j);
            }

            newInst.put(0, sum);
            newInst.setClassValue(instanc.classValue());

            newData.add(newInst);
        }
        return newData;
    }

    /*check the number of features selected
     */
    public static int noFeature(double[] features) {

        int sum = 0;

        for (int i = 0; i < features.length; i++) {
            if (features[i] > 0.5) {
                sum++;
            }
        }

        return sum;
    }
}
