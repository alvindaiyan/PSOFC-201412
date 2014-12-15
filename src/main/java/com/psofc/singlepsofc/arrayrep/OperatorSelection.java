package com.psofc.singlepsofc.arrayrep;

import java.util.List;

public class OperatorSelection {


	public static char[] getOpFromArray(List<Double> features) {
		char[] operators = new char[(features.size() -1) / 2];
		operators = initial(operators);
		int o = 0;
        for (int i = 1; i < features.size(); i+=2) {
        	if(features.get(i - 1) >= 0.5){
	            if(features.get(i) >= 0.0 && features.get(i) < 0.25){
	            	operators[o++] = '+';
	            } else if(features.get(i) >= 0.25 && features.get(i) < 0.5){
	            	operators[o++] = '-';
	            } else if(features.get(i) >= 0.5 && features.get(i) < 0.75){
	            	operators[o++] = '*';
	            } else if(features.get(i) >= 0.75 && features.get(i) <= 1.0){
	            	operators[o++] = '/';
	            }
        	}
        }
		return operators;
	}

	public static char[] getOpFromArray(double[] features) {
		char[] operators = new char[(features.length - 1) / 2];
		operators = initial(operators);
		int o = 0;
        for (int i = 1; i < features.length; i+=2) {
        	if(features[i - 1] >= 0.5){
	            if(features[i] >= 0 && features[i] < 0.25){
	            	operators[o++] = '+';
	            } else if(features[i] >= 0.25 && features[i] < 0.5){
	            	operators[o++] = '-';
	            } else if(features[i] >= 0.5 && features[i] < 0.75){
	            	operators[o++] = '*';
	            } else if(features[i] >= 0.75 && features[i] <= 1.0){
	            	operators[o++] = '/';
	            }
        	}
        }
		return operators;
	}

	public static double[] getFeatures(double[] position){
		double[] features = new double[(position.length + 1)/2];
		int p = 0;
		for (int i = 0; i < position.length; i += 2) {
			features[p++] = position[i];
		}

		return features;
	}

	private static char[] initial(char[] op){
		for(int i = 0; i < op.length; i++){
			op[i] = 'a';
		}

		return op;
	}

}
