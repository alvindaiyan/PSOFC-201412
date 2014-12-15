/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 *
 * @author xuebing
 */
public class Particle {

    private List<Double> _position = new ArrayList<Double>();
    private List<Double> _velocity = new ArrayList<Double>();
    private double _fitness;
    private List<Double> _personal_position = new ArrayList<Double>();
    private double _personal_fitness;
    private List<Double> _neighborhood_position = new ArrayList<Double>();
    private double _neighborhood_fitness;
    private double _inertia;
    private double _c1, _c2, _c3;
    private Random _r1 = RandomBing.Create(), _r2 = RandomBing.Create(), _rand = RandomBing.Create();// ,_r3 = RandomBing.Create(); //
    private double _max_position, _min_position;
    private Problem _problem;

    /**
     *
     */
    public Particle() {
    }

    /**
     *
     * @param size
     */
    public void setSize(int size) {
        _position.clear();
        _velocity.clear();
        _personal_position.clear();
        _neighborhood_position.clear();
        for (int i = 0; i < size; ++i) {
            _position.add(0.0);
            _velocity.add(0.0);
            _personal_position.add(0.0);
            _neighborhood_position.add(0.0);
        }
    }

    /**
     *
     * @return
     */
    public int getSize() {
        return _position.size();
    }

    /**
     *
     * @param index
     * @param value
     */
    public void setPosition(int index, double value) {
        this._position.set(index, value);
    }

    /**
     *
     * @param index
     * @return
     */
    public double getPosition(int index) {
        return _position.get(index);
    }

    /**
     *
     * @return
     */
    public List<Double> getPosition() {
        return _position;
    }

    /**
     *
     * @return
     */
    public List<Double> getNeighborhoodPosition() {
        return _neighborhood_position;
    }

    /**
     *
     * @return
     */
    public List<Double> getPersonalPosition() {
        return _personal_position;
    }

    /**
     *
     * @param index
     * @param value
     */
    public void setVelocity(int index, double value) {
        _velocity.set(index, value);
    }

    /**
     *
     * @param index
     * @return
     */
    public double getVelocity(int index) {
        return _velocity.get(index);
    }

    /**
     *
     * @return
     */
    public double getFitness() {
        return _fitness;
    }

    /**
     *
     * @param fitness
     */
    public void setFitness(double fitness) {
        this._fitness = fitness;
    }

    /**
     *
     * @param index
     * @param value
     */
    public void setPersonalPosition(int index, double value) {
        _personal_position.set(index, value);
    }

    /**
     *
     * @param index
     * @return
     */
    public double getPersonalPosition(int index) {
        return _personal_position.get(index);
    }

    /**
     *
     * @return
     */
    public double getPersonalFitness() {
        return _personal_fitness;
    }

    /**
     *
     * @param fitness_best_personal
     */
    public void setPersonalFitness(double fitness_best_personal) {
        _personal_fitness = fitness_best_personal;
    }

    /**
     *
     * @param index
     * @param value
     */
    public void setNeighborhoodPosition(int index, double value) {
        this._neighborhood_position.set(index, value);
    }

    /**
     *
     * @param index
     * @return
     */
    public double getNeighborhoodPosition(int index) {
        return _neighborhood_position.get(index);
    }

    /**
     *
     * @return
     */
    public double getNeighborhoodFitness() {
        return _neighborhood_fitness;
    }

    /**
     *
     * @param fitness_best_neighbor
     */
    public void setNeighborhoodFitness(double fitness_best_neighbor) {
        this._neighborhood_fitness = fitness_best_neighbor;
    }

    /**
     *
     * @return
     */
    public double getInertia() {
        return _inertia;
    }

    /**
     *
     * @param inertia
     */
    public void setInertia(double inertia) {
        this._inertia = inertia;
    }

    /**
     *
     * @return
     */
    public double getC1() {
        return _c1;
    }

    /**
     *
     * @param c1
     */
    public void setC1(double c1) {
        this._c1 = c1;
    }

    /**
     *
     * @return
     */
    public double getC2() {
        return _c2;
    }

    /**
     *
     * @param c2
     */
    public void setC2(double c2) {
        this._c2 = c2;
    }

    /**
     *
     * @return
     */
    public Random getR1() {
        return _r1;
    }

    /**
     *
     * @return
     */
    public Random getR2() {
        return _r2;
    }

    /**
     *
     * @return
     */
    public double getMaxPosition() {
        return _max_position;
    }

    /**
     *
     * @param max_position
     */
    public void setMaxPosition(double max_position) {
        this._max_position = max_position;
    }

    /**
     *
     * @return
     */
    public double getMinPosition() {
        return _min_position;
    }

    /**
     *
     * @param min_position
     */
    public void setMinPosition(double min_position) {
        this._min_position = min_position;
    }

    /**
     *
     * @param w
     */
    public void updateVelocity(double w) {
        for (int i = 0; i < getSize(); ++i) {
            double v_i = w * getVelocity(i);//v_i=0; //
            v_i += getC1() * getR1().nextDouble() * (getPersonalPosition(i) - getPosition(i));
            v_i += getC2() * getR2().nextDouble() * (getNeighborhoodPosition(i) - getPosition(i));
            setVelocity(i, v_i);
        }
    }

    /**
     * */
    public void updatePosition() {
        for (int i = 0; i < getSize(); ++i) {
//  Binary PSO****************************************************
//            double p_i = 0.0;
//            double sig = 1 / (1 + Math.exp(-getVelocity(i)));
//            if (getRand().nextDouble() < sig) {
//                p_i = 1.0;
//            } else {
//                p_i = 0.0;
//            }


//// //                    bs = 1 / (1 + Math.pow(Math.E, -getVelocity(i)));

//  Continuous PSO****************************************************
            double p_i = getPosition(i) + getVelocity(i);

            if (p_i > getMaxPosition()) {
                p_i = getMaxPosition();
            }
            if (p_i < getMinPosition()) {
                p_i = getMinPosition();
            }

//  *******************************************************************
            setPosition(i, p_i);
        }
    }

    public void updatePositionCro() {

        for (int i = 0; i < getSize(); ++i) {
            if (getFitness() == getNeighborhoodFitness()) {
                setPosition(i, getNeighborhoodPosition(i));
            } else {
                double prob = (getPersonalFitness() - getFitness()) / (getNeighborhoodFitness() - getFitness());
                if (prob < getR1().nextDouble()) {
                    setPosition(i, getNeighborhoodPosition(i));
                } else {
                    setPosition(i, getPersonalPosition(i));
                }
            }
        }
    }

    /**
     * @return the _problem1
     */
    public Problem getProblem() {
        return _problem;
    }

    /**
     * @param problem
     */
    public void setProblem(Problem problem) {
        this._problem = problem;
    }

    /**
     * @return the _c3
     */
    public double getC3() {
        return _c3;
    }

    /**
     * @param c3 the _c3 to set
     */
    public void setC3(double c3) {
        this._c3 = c3;
    }

    /**
     * @return the _r3
     */
//    public Random getR3() {
//        return _r3;
//    }
//
//    /**
//     * @param r3 the _r3 to set
//     */
//    public void setR3(Random r3) {
//        this._r3 = r3;
//    }
    /**
     * @return the _rand
     */
    public Random getRand() {
        return _rand;
    }

    /**
     * @param rand the _rand to set
     */
    public void setRand(Random rand) {
        this._rand = rand;
    }
    /**
     * @return the _r3
     */
//    public Random getR3() {
//        return _r3;
//    }
//
//    /**
//     * @param r3 the _r3 to set
//     */
//    public void setR3(Random r3) {
//        this._r3 = r3;
//    }
}
