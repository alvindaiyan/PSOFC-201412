/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc;


import com.psofc.util.NewMath;

/**
 *
 * @author xuebing
 */
public final class TopologyRing extends Topology {

    private int _neighbors = 2;

    /**
     *
     */
    public TopologyRing() {
    }

    /**
     *
     * @param n
     */
    public TopologyRing(int n) {
        setNeighbors(n);
        System.out.println("Topology topology = new TopologyRing(n);  n=" + n);
    }

    /**
     *
     * @param s
     */
    public void share(Swarm s) {

        for (int i = 0; i < s.numberOfParticles(); ++i) {

            Particle p_i = s.getParticle(i);
//            Particle best_neighbor = null;
            Particle best_neighbor = p_i;
            double best_fitness = s.getProblem().getWorstFitness();
//            double best_fitness = p_i.getNeighborhoodFitness();  // This does not work !!! be careful

            for (int j = -getNeighbors() / 2; j <= getNeighbors() / 2; ++j) {
                int neigherID = NewMath.ModEuclidean(i + j, s.numberOfParticles());
                if (s.getProblem().isBetter(s.getParticle(neigherID).getPersonalFitness(), best_fitness)) {
                    best_neighbor = s.getParticle(neigherID);
                    best_fitness = best_neighbor.getPersonalFitness();
                }
            }
            p_i.setNeighborhoodFitness(best_fitness);
            for (int n = 0; n < p_i.getSize(); ++n) {
                p_i.setNeighborhoodPosition(n, best_neighbor.getPersonalPosition(n));

            }
        }
    }

    /**
     *
     * @return
     */
    public int getNeighbors() {
        return _neighbors;
    }

    /**
     *
     * @param neighbors
     */
    public void setNeighbors(int neighbors) {
        this._neighbors = neighbors;
    }
}
//   int n = 10;
//        int nei = 4;
//
//        for (int j = 0; j < 10; ++j) {
//            System.out.print(j + ":");
//
//            for (int p =-nei/2; p <= nei/2 ; ++p) {
//
//                System.out.print(Math.ModEuclidean(p + j, n) + " ; ");
//            }
//            System.out.println();
//        }
//        System.out.print((1/2)*nei);
//        System.exit(32);
//0:8 ; 9 ; 0 ; 1 ; 2 ;
//1:9 ; 0 ; 1 ; 2 ; 3 ;
//2:0 ; 1 ; 2 ; 3 ; 4 ;
//3:1 ; 2 ; 3 ; 4 ; 5 ;
//4:2 ; 3 ; 4 ; 5 ; 6 ;
//5:3 ; 4 ; 5 ; 6 ; 7 ;
//6:4 ; 5 ; 6 ; 7 ; 8 ;
//7:5 ; 6 ; 7 ; 8 ; 9 ;
//8:6 ; 7 ; 8 ; 9 ; 0 ;
//9:7 ; 8 ; 9 ; 0 ; 1 ;

