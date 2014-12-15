/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc.util.ttest;

import jsc.independentsamples.TwoSampleTtest;
import jsc.tests.H1;
import org.apache.commons.math.MathException;
import org.apache.commons.math.stat.inference.TTestImpl;

/**
 *
 * @author xuebing
 */
public class Ttest {

    public static String TtestString(double[] s1, double[] s2) throws IllegalArgumentException, MathException {
        TTestImpl Ttest = new TTestImpl();
        String strTest = null;
        double temP = Ttest.pairedTTest(s1, s2);
        if (temP > 0.05) {
            strTest = "=";
        } else {
            if (Ttest.pairedTTest(s2, s1) < 0.05) {
                strTest = "-";
            } else {
                strTest = "+";
            }

        }
        return strTest;
    }

    public static double TtestDouble(double[] s1, double[] s2) throws IllegalArgumentException, MathException {
        TTestImpl Ttest = new TTestImpl();
        double pValue = 0.0;
        double temP = Ttest.pairedTTest(s1, s2);

        if (Helper.mean(s1) < Helper.mean(s2)) {
            pValue = 0.5 * temP;

        } else {
            pValue = 1.0 - 0.5 * temP;
        }

        System.out.println("");
        System.out.println("pValue  " + Ttest.pairedTTest(s1, s2));
        return pValue;
    }
 

    public static String TtestBingNew(double[] s1, double[] s2) {
        double pValue = 0.0;
        double sigLevel= 0.05;
        String strTest = null;

        if (Helper.ifSameEleArray(s1) & Helper.ifSameEleArray(s2)) {
            if (s1[0] == s2[0]) {
                strTest = "=";
            } else if (s1[0] < s2[0]) {
                strTest = "+";
            } else {
                strTest = "-";
            }
        } else {
            TwoSampleTtest w0 = new TwoSampleTtest(s1, s2, H1.NOT_EQUAL, false);
            if (w0.getSP() > sigLevel) {
                strTest = "=";
                pValue = w0.getSP();
//            System.out.println("w0.getSP()" + w0.getSP());
            } else {
                TwoSampleTtest w1 = new TwoSampleTtest(s1, s2, H1.LESS_THAN, false);
                if (w1.getSP() <= sigLevel) {
                    strTest = "+";
                    pValue = w1.getSP();
//                System.out.println("w1.getSP()" + w1.getSP());
                } else {
                    TwoSampleTtest w2 = new TwoSampleTtest(s1, s2, H1.GREATER_THAN, false);
                    if (w2.getSP() < sigLevel) {
                        strTest = "-";
                        pValue = w2.getSP();
//                    System.out.println("w2.getSP()" + w2.getSP());
                    } else {
                        strTest = "!!!!!!!";
                    }
                }
            }
        }
//        return pValue;
        return strTest;
    }
}
