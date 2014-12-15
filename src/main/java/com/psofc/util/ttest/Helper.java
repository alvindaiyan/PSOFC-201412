/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc.util.ttest;


/**
 *
 * @author xuebing
 */
public class Helper {

    public static double[] sameEleArray(int n, double value) {
        double[] array = new double[n];
        for (int i = 0; i < n; i++) {
            array[i] = value;
        }
        return array;
    }


    public static  boolean ifSameEleArray(double[] acc) {
        boolean same = true;
        double ini = acc[0];
        for (int i = 0; i < acc.length; i++) {
            if (!(ini == acc[i])) {
                same = false;
                break;
            }
        }
        return same;
    }

// get mean
    public static double mean(double[] array) {
        double sum = 0.0;  // sum of all the elements
        if (array.length == 0) {
            System.out.println("This array with no elements in it !!!!");
        } else {
            for (int i = 0; i < array.length; i++) {
                sum += array[i];
            }
        }
        return sum / array.length;
    }//end method mean


}