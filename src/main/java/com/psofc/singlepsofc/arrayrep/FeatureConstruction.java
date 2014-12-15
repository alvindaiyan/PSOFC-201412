/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc.singlepsofc.arrayrep;

import java.io.IOException;
import java.util.List;

import com.psofc.HelpDataset;
import com.psofc.Problem;
import com.psofc.util.ClassifierYan;
import com.psofc.util.FCFunction;
import com.psofc.util.NewMath;


import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;

/**
 *
 * @author xuebing
 */
// feature construction using BPSO, minimisation problem
public class FeatureConstruction extends Problem {

    public FeatureConstruction() throws IOException {
        System.out.println("FeatureConstruction ");
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

//        for(int i = 0; i < features.length; i++){
//        	System.out.print(features[i] + " ");
//        }
//        System.out.println();

        int sum = FCFunction.noFeature(features);

        if (sum != 0) {

            Dataset[] foldsTem = new Dataset[getFoldsTrain().length];

            char[] operators = new char[(position.size() - 1) / 2];  // temperory, this need to according the results of the selected operators

            operators = OperatorSelection.getOpFromArray(position);


            for (int f = 0; f < getFoldsTrain().length; f++) {

                Dataset dataTem = new DefaultDataset();

                dataTem = getFoldsTrain()[f].copy();

                /* remove unselected feature first
                 */
                dataTem = HelpDataset.removeFeatures(dataTem, features);

                /* change dataset using constructed feature
                 */
                dataTem = FCFunction.calConstructFeaBing(dataTem, operators);

                foldsTem[f] = dataTem;
            }

            double accuracy = 0.0;

            /* getNumFols cross validation, return the average of accuracies;*/
            for (int f = 0; f < getNumFolds(); f++) {
                Dataset testTem = foldsTem[f];
                Dataset trainTem = new DefaultDataset();
                for (int j = 0; j < getNumFolds(); j++) {
                    if (j != f) {
                        trainTem.addAll(foldsTem[j]);
                    }
                }
                accuracy += this.classify(trainTem, testTem);
            }
            double error = 1.0 - accuracy / (double) getNumFolds();
//            System.out.println("error = " + error);
            return error;

        } else {
            return 10000000;  // if maximisation problem, then use the -1000000000 whatever the worst solution;
        }
    }



    public double classify(Dataset training, Dataset testing){
    	return ClassifierYan.classify(training, testing);
    }
}
