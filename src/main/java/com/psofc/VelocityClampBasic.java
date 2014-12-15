/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package com.psofc;

/**
 *
 * @author xuebing
 */
public class VelocityClampBasic extends VelocityClamp {

    public VelocityClampBasic() {
    }

    public void clamp(Particle p, double MaxVelocity, double MinVelocity) {
        for (int i = 0; i < p.getSize(); ++i) {
            //This should be the absolute value... TODO
            if (p.getVelocity(i) > MaxVelocity) {
                p.setVelocity(i, MaxVelocity);
            } else if (p.getVelocity(i) < MinVelocity) {
                p.setVelocity(i, MinVelocity);
            }
        }
    }
}
