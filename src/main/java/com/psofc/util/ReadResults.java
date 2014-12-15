/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc.util;

import java.io.*;
import java.util.ArrayList;
import java.util.List;

//import java.util.*;
//import java.lang.Math;
/**
 *
 * @author xuebing
 */
public class ReadResults {

    public static String[] read2array(String filename, String[] strTem) throws IOException {
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
            int i = 0;
            String line = null;
            while ((line = bufferedReader.readLine()) != null && i < strTem.length) {
                strTem[i] = line;
                i++;
            }
            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return strTem;
    }

    public static ArrayList<String> readAll2list(String filename) throws IOException {
        ArrayList<String> tem = new ArrayList<String>();
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));
            int i = 0;
            String line = null;
            while ((line = bufferedReader.readLine()) != null) {
                tem.add(line);
                i++;
            }
            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return tem;
    }

    public static String[] readSomLine2array(String filename, String[] strTem, String search4) throws IOException {
        try {
            BufferedReader r = new BufferedReader(new FileReader(filename));
            String line = null;
            int i = 0;

            for (int j = 0; j < 1000; j++) {
                line = r.readLine();
                if (line.equals(search4)) {
                    System.out.println("yes here !!!");
                    break;
                }
                if (j == 999) {
                    System.out.println("Something wrong, could not find the target line");
                }
            }

            while ((line = r.readLine()) != null && i < strTem.length) {
                strTem[i] = line;
                i++;
            }
            r.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return strTem;
    }

    public static String read1Line(String filename) throws IOException {
        String line = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

            line = bufferedReader.readLine();

            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return line;
    }

    public static String read1stLine(String filename) throws IOException {
        String line = null;
        try {
            BufferedReader bufferedReader = new BufferedReader(new FileReader(filename));

            line = bufferedReader.readLine();

            bufferedReader.close();
//            in.close();
        } catch (Exception e) {//Catch exception if any
            e.printStackTrace();
        }
        return line;
    }

    public static double[] str2DoubleArray(String[] array) {
        double[] newArray = new double[array.length];
        for (int i = 0; i < array.length; i++) {
//            System.out.println(array[i]);
            newArray[i] = Double.valueOf(array[i]);
        }
        return newArray;
    }
}
