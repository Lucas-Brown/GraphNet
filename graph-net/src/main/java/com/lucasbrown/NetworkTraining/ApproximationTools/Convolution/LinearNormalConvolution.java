package com.lucasbrown.NetworkTraining.ApproximationTools.Convolution;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.NormalDistribution;

public class LinearNormalConvolution extends GenericConvolution {

    private NormalDistribution dist;
    private double weight;

    public LinearNormalConvolution(NormalDistribution dist, double weight){
        super(dist.getActivatedDistribution(ActivationFunction.LINEAR, weight));
        this.dist = dist;
        this.weight = weight;
    }

    @Override
    public IConvolution convolveWith(IConvolution g) {
        if(g instanceof LinearNormalConvolution){
            return convolveWithLinear((LinearNormalConvolution) g);
        }

        return super.convolveWith(g);
    }

    private LinearNormalConvolution convolveWithLinear(LinearNormalConvolution lbc){
        double weighted_mean = weight*dist.getMean() + lbc.weight*lbc.dist.getMean();
        double weighted_variance = weight*weight*dist.getVariance()*dist.getVariance();
        weighted_variance += lbc.weight*lbc.weight*lbc.dist.getVariance()*lbc.dist.getVariance();
        weighted_variance = Math.sqrt(2*weighted_variance);
        return new LinearNormalConvolution(new NormalDistribution(weighted_mean, weighted_variance), 1);
    }
    
}
