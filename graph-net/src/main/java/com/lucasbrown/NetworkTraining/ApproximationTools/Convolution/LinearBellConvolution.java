package com.lucasbrown.NetworkTraining.ApproximationTools.Convolution;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveFilter;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;

public class LinearBellConvolution extends GenericConvolution {

    private BellCurveFilter bcd;
    private double weight;

    public LinearBellConvolution(BellCurveFilter bcd, double weight){
        super(bcd.getActivatedDistribution(ActivationFunction.LINEAR, weight));
        this.bcd = bcd;
        this.weight = weight;
    }

    @Override
    public IConvolution convolveWith(IConvolution g) {
        if(g instanceof LinearBellConvolution){
            return convolveWithLinear((LinearBellConvolution) g);
        }

        return super.convolveWith(g);
    }

    private LinearBellConvolution convolveWithLinear(LinearBellConvolution lbc){
        double weighted_mean = weight*bcd.getMean() + lbc.weight*lbc.bcd.getMean();
        double weighted_variance = weight*weight*bcd.getVariance()*bcd.getVariance();
        weighted_variance += lbc.weight*lbc.weight*lbc.bcd.getVariance()*lbc.bcd.getVariance();
        weighted_variance = Math.sqrt(2*weighted_variance);
        return new LinearBellConvolution(new BellCurveFilter(weighted_mean, weighted_variance), 1);
    }
    
}
