/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc;

import java.util.HashSet;
import java.util.Set;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

/**
 *
 * @author xuebing
 */
public class HelpDataset {

    /*
     * Removes the element from the vector that is farthest from the supplied
     * element.
     */

    public static Dataset RemoveFarthest(double[] position, Dataset dataTem, Problem problem) {
        //*     remove other features
        Set<Integer> set = new HashSet<Integer>();
        if (problem.getBinaryContin()) {
            for (int d = 0; d < position.length; d++) {
                if (position[d] == 0.0) {
                    set.add(d);
                }
            }

        } else {
            for (int d = 0; d < position.length; d++) {
                if (position[d] < problem.getThreshold()) {
                    set.add(d);
                }
            }
        }

        for (int i = 0; i < dataTem.size(); i++) {
            dataTem.get(i).removeAttributes(set);
        }

        return dataTem;

    }

    public static Dataset removeFeatures(Dataset data, double[] features) {
//
        Dataset n = new DefaultDataset();
        int count = 0;
        
        
        
        for (double d : features) {
            if (d >= 0.5) {
                count++;
            }
        }

        for (Instance instance : data) {
//            System.out.println("features.length  " + instance.noAttributes());
            double[] f = new double[count];
            int j = 0;
//            System.out.println("features.length  " + features.length);
            for (int i = 0; i < features.length; i++) {
                if (features[i] >= 0.5) {
                    f[j++] = instance.get(i);
                }
            }

            Instance inst = new DenseInstance(f, instance.classValue());
            n.add(inst);
        }

        return n;
    }

        public static int sizeSubset(Particle particle, Problem problem) {
        int size = 0;
        if (problem.getBinaryContin()) {
            for (int i = 0; i < particle.getSize(); i++) {
                if (particle.getPosition().get(i) == 1.0) {
                    size++;
                }
            }
        } else {
            for (int i = 0; i < particle.getSize(); i++) {
                if (particle.getPosition().get(i) >= problem.getThreshold()) {
                    size++;
                }
            }
        }

        return size;
    }

}
