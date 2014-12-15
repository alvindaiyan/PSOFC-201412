/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc;

import java.util.List;
import net.sf.javaml.core.Dataset;

/**
 *
 * @author xuebing
 */
public abstract class Problem {

    private boolean _minimization = true;
    private double _threshold;
    private double _max_domain, _min_domain;
    private double _max_velocity, _min_velocity;
    private int _dimension;
    private int _numFolds;
    private double fullTrainEr;
    private MyClassifier _myclassifier;
    private Dataset _training;
    private Dataset[] _foldsTrain;
    private Boolean binaryContin;

    public Problem() {
    }

    public boolean isMinimization() {
        return _minimization;
    }

    public void setMinimization(boolean minimization) {
        this._minimization = minimization;
    }

    public double getMaxDomain() {
        return _max_domain;
    }

    public void setMaxDomain(double max_domain) {
        this._max_domain = max_domain;
    }

    public double getMinDomain() {
        return _min_domain;
    }

    public void setMinDomain(double min_domain) {
        this._min_domain = min_domain;
    }

    /**
     * @return the _max_velocity
     */
    public double getMaxVelocity() {
        return _max_velocity;
    }

    /**
     * @param max_velocity the _max_velocity to set
     */
    public void setMaxVelocity(double max_velocity) {
        this._max_velocity = max_velocity;
    }

    /**
     * @return the _min_velocity
     */
    public double getMinVelocity() {
        return _min_velocity;
    }

    /**
     * @param min_velocity the _min_velocity to set
     */
    public void setMinVelocity(double min_velocity) {
        this._min_velocity = min_velocity;
    }

    public boolean isBetter(double fitness_a, double fitness_b) {
        return isMinimization() ? fitness_a < fitness_b : fitness_a > fitness_b;
    }

    /**
     * @return the _threshold
     */
    public double getThreshold() {
        return _threshold;
    }

    /**
     * @param threshold the _threshold to set
     */
    public void setThreshold(double threshold) {
        this._threshold = threshold;
    }

    public int getDimension() {
        return _dimension;
    }

    public void setDimension(int dimension) {
        this._dimension = dimension;
    }

    /**
     * @return the _training
     */
    public Dataset getTraining() {
        return _training;
    }

    /**
     * @param training the _training to set
     */
    public void setTraining(Dataset training) {
        this._training = training;
    }

    public double getWorstFitness() {
        return isMinimization() ? Double.POSITIVE_INFINITY : Double.NEGATIVE_INFINITY;
    }

    public abstract double fitness(List<Double> position);

    /**
     * @return the _numFolds
     */
    public int getNumFolds() {
        return _numFolds;
    }

    /**
     * @param numFoldsCV the _numFolds to set
     */
    public void setNumFolds(int numFoldsCV) {
        this._numFolds = numFoldsCV;
    }

    /**
     * @return the _foldsTrain
     */
    public Dataset[] getFoldsTrain() {
        return _foldsTrain;
    }

    /**
     * @param foldsTrain the _foldsTrain to set
     */
    public void setFoldsTrain(Dataset[] foldsTrain) {
        this._foldsTrain = foldsTrain;
    }

    /**
     * @return the fullTrainEr
     */
    public double getFullTrainEr() {
        return fullTrainEr;
    }

    /**
     * @param fullTrainEr the fullTrainEr to set
     */
    public void setFullTrainEr(double fullTrainEr) {
        this.fullTrainEr = fullTrainEr;
    }

    /**
     * @return the binaryContin
     */
    public Boolean getBinaryContin() {
        return binaryContin;
    }

    /**
     * @param binaryContin the binaryContin to set
     */
    public void setBinaryContin(Boolean binaryContin) {
        this.binaryContin = binaryContin;
    }

    /**
     * @return the _myclassifier
     */
    public MyClassifier getMyclassifier() {
        return _myclassifier;
    }

    /**
     * @param myclassifier the _myclassifier to set
     */
    public void setMyclassifier(MyClassifier myclassifier) {
        this._myclassifier = myclassifier;
    }
}
