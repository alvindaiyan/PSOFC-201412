package com.psofc.util;


import java.util.Random;

import com.psofc.MyClassifier;


import net.sf.javaml.core.Dataset;


public class ClassifierYan {


//    static final double THRESHOLD = 0.0;
//    public static double classify(Dataset training, Dataset testing){
//    	// initial the threshold and two classes
//    	Object[] clzz = new Object[2];
//    	int index = 0;
//    	for(Object o : training.classes()){
//    		clzz[index] = o;
//    		index++;
//    	}
//    	Object cls_0 =clzz[0];
//    	int cls0 = 0;
//    	boolean isPositive_cls0 = false;
//    	int cls1 = 0;
//    	for(Instance instance : training){
//    		if(instance.value(0) > THRESHOLD){
//    				if(instance.classValue().equals(cls_0)){
//    					cls0++;
//    				}else {cls1++;}
//    		}else {
//    			if(instance.classValue().equals(cls_0)){
//    				cls1++;
//    			}else {cls0++;}
//    		}
//    	}
//
//    	if(cls0 > cls1){
//    		// positive is cls_0
//    		isPositive_cls0 = true;
//    	}// negative is cls_0
//
//    	Dataset dataForClassification = testing;
//
//        Map<Object, PerformanceMeasure> out = new HashMap<Object, PerformanceMeasure>();
//        for (Object o : training.classes()) {
//            out.put(o, new PerformanceMeasure());
//        }
//
//        for (Instance instance : dataForClassification){
//        	Object prediction = null;	// need to identify
//        	if(isPositive_cls0){
//        		// cls0 is on the positive side
//        		if(instance.value(0) > THRESHOLD ){prediction = cls_0;}
//        		else {prediction = clzz[1];}
//        	}else{
//        		if(instance.value(0) <= THRESHOLD ){prediction = cls_0;}
//        		else {prediction = clzz[1];}
//        	}
//
//        	if(instance.classValue().equals(prediction)){
//        		for(Object o : out.keySet()){
//        			if (o.equals(instance.classValue())) {
//        				out.get(o).tp++;
//        			} else{
//        				out.get(o).tn++;
//        			}
//        		}
//        	} else {
//        		for (Object o : out.keySet()){
//        			if (prediction.equals(o)){
//        				out.get(o).fp++;
//        			} else if (o.equals(instance.classValue())) {
//        				out.get(o).fn++;
//        			} else {
//        				out.get(o).tn++;
//        			}
//        		}
//        	}
//        }
//
//
//        double tp = 0.0, tn = 0.0;
//        double Accuracy = 0.0;
//        for(Object o : out.keySet()){
//        	tp += out.get(o).tp;
//        	tn += out.get(o).tn;
//        }
//        Accuracy = (tn + tp) / (double) (out.size() * dataForClassification.size());
//
//
//    	return Accuracy;
//    }


    public static double classify(Dataset training, Dataset testing){
    	MyClassifier mc = new MyClassifier(new Random(1));
        mc.ClassifierDT();

        double accuracy;

//      for(int rm = 0; rm < testing.size(); rm++){
//		Instance line = testing.get(rm);
//		System.out.println("value: " + line.get(0) + " , class: " + line.classValue());
//		}


//        System.out.println("new start: " + training.size() + " " + testing.size());
//        try{
        	accuracy = mc.classify(training, testing);
////        }catch(Exception e){
//        	accuracy = 10000000;
//          	e.printStackTrace();
//        }catch(Error e){
//          	accuracy = 10000000;
//          	e.printStackTrace();
//        }

    	return accuracy;
    }
}
