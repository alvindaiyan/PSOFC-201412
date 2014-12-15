package com.psofc.util;

import java.util.HashMap;
import java.util.Map;
import java.util.Set;

import net.sf.javaml.classification.AbstractClassifier;
import net.sf.javaml.core.Dataset;
import net.sf.javaml.core.Instance;
import net.sf.javaml.core.exception.TrainingRequiredException;
import net.sf.javaml.distance.DistanceMeasure;
import net.sf.javaml.distance.EuclideanDistance;

public class KNearestNeighbors extends AbstractClassifier {

	private int k;
	private EuclideanDistance dm;
	private Dataset training;

	public KNearestNeighbors(int k) {
		this(k, new EuclideanDistance());
	}

	public KNearestNeighbors(int k, EuclideanDistance euclideanDistance) {
		this.k = k;
		this.dm = euclideanDistance;

	}

	@Override
	public void buildClassifier(Dataset arg0) {
		this.training = arg0;

	}

	@Override
	public Map<Object, Double> classDistribution(Instance instance) {
		if (training == null) {
			throw new TrainingRequiredException();
		}

		Set<Instance> neighbors = kNearest(k, instance, dm, training);

		if (neighbors.size() == 0) {
			System.err.println("neighbors is null! " + dm.getMaxValue());
		}

		Map<Object, Double> out = new HashMap<Object, Double>();

		for (Object o : training.classes()) {
			out.put(o, 0.0);
		}

		for (Instance i : neighbors) {
			out.put(i.classValue(), out.get(i.classValue()) + 1);
		}

		double min = k;
		double max = 0;

		for (Object key : out.keySet()) {
			double val = out.get(key);
			if (val > max) {
				max = val;
			}
			if (val < min) {
				min = val;
			}
		}

		if (max != min) {
			for (Object key : out.keySet()) {
				out.put(key, (out.get(key) - min) / (max - min));
			}
		}

		return out;
	}

	private Set<Instance> kNearest(int k, Instance inst, DistanceMeasure dmk, Dataset dt) {
		Map<Instance, Double> closest = new HashMap<Instance, Double>();
		double max = dmk.getMaxValue();
		// System.out.println("max---" + dm.getMaxValue());
		for (Instance tmp : dt) {
			double d = dmk.measure(inst, tmp);
			// System.out.println("ddddd:  " + d);
			// System.out.println("inst: " + inst + "   tmp:   " + tmp);
			if (d == 0 && !inst.equals(tmp)) {
				// System.out.println("Same values and different classes");
			} // segmentation.data, print "Same values and different classes";
			else if (d == 0 && inst.equals(tmp)) {
				// System.out.println("...");
			}
			// if (dm.compare(d, max) && !inst.equals(tmp)) {
			if (dmk.compare(d, max)) {
				// System.out.println("compare(d, max), d  " + d + "  max   " +
				// max);
				closest.put(tmp, d);
				// System.out.println("closest.size():  " + closest.size());
				if (closest.size() > k) {
					max = removeFarthest(closest, dmk);
				}
			}
		}
		return closest.keySet();

	}


    private double removeFarthest(Map<Instance, Double> vector, DistanceMeasure dm) {
        Instance tmp = null;// ; = vector.get(0);
        double max = dm.getMinValue();
//        System.out.println("minvalue:" + max);
        for (Instance inst : vector.keySet()) {
            double d = vector.get(inst);
            if (dm.compare(max, d)) {
                max = d;
                tmp = inst;
            }
            // System.out.println("d="+d+"\t"+max);
        }
        vector.remove(tmp);
        return max;

    }

}
