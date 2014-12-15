package com.psofc.util;


import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.DefaultDataset;
import net.sf.javaml.core.DenseInstance;
import net.sf.javaml.core.Instance;

/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
/**
 *
 * @author xuebing
 */
public class dataNomarlise {

    public static Dataset DataNormalisation(Dataset train) {

        Dataset oldData = train.copy();
        Dataset newData = new DefaultDataset();

        double normaliseToMax = 1;
        double normaliseToMin = 0;

        double[] max = new double[train.noAttributes()];
        double[] min = new double[train.noAttributes()];

        for (int f = 0; f < train.noAttributes(); f++) {
            double maxAtt = train.get(1).get(f);
            double minAtt = train.get(1).get(f);
            for (int in = 0; in < train.size(); in++) {
                if (maxAtt < train.get(in).get(f)) {
                    maxAtt = train.get(in).get(f);
                } else if (minAtt > train.get(in).get(f)) {
                    minAtt = train.get(in).get(f);
                }
            }
            max[f] = maxAtt;
            min[f] = minAtt;
        }
//        System.out.println("try  7  ----  "+NewMath.Scale(0.11, 2.87, 1.05, 0, 1));
        for (int in = 0; in < oldData.size(); in++) {
            Instance tem = new DenseInstance(oldData.noAttributes());
            for (int f = 0; f < oldData.noAttributes(); f++) {
                tem.put(f, Scale(min[f], max[f], oldData.get(in).get(f), normaliseToMin, normaliseToMax));
            }
            tem.setClassValue(oldData.get(in).classValue());
            newData.add(tem);
        }
        return newData;
    }

    public static double Scale(double src_min, double src_max, double value, double target_min, double target_max) {
        if (src_max == src_min) {
            return (1 / 2) * (target_max - target_min) + target_min;
        } else {
            return (target_max - target_min) * (value - src_min) / (src_max - src_min) + target_min;
        }
    }
}
