package com.psofc.singlepsofc.pair;

public class OperatorSelection {

	public static char[] getOp(double[] features){
		char[] operators = new char[features.length - 1];
		int o = 0;
        for (int i = 0; i < operators.length; i++) {
            if(features[i] >= 0.5 && features[i] < 0.625){
            	operators[o++] = '+';
            } else if(features[i] >= 0.625 && features[i] < 0.75){
            	operators[o++] = '-';
            } else if(features[i] >= 0.75 && features[i] < 0.875){
            	operators[o++] = '*';
            } else if(features[i] >= 0.875 && features[i] <= 1.0){
            	operators[o++] = '/';
            }

//            if(features[i] == 0.5015303720914831){System.out.println("operators: " + operators[o-1]);}
        }
		return operators;
	}

}
