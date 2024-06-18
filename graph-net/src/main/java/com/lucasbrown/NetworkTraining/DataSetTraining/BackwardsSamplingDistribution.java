package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.DoubleFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.IntegralTransformations;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.GenericConvolution;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.IConvolution;

import jsat.math.integration.Trapezoidal;

public abstract class BackwardsSamplingDistribution implements ITrainableDistribution {
    
    protected Random rng;
    private static final int TRAP_COUNT = 100;

    public BackwardsSamplingDistribution(Random random){
        rng = random;
    }

    public abstract double sample();

    /**
     * Get the mean value of a distribution whose underlying data has undergone the
     * transformation of the activator
     * 
     * @param activator
     * @param w
     * @return
     */
    public double getMeanOfAppliedActivation(ActivationFunction activator, double w) {
        DoubleUnaryOperator integrand = t -> IntegralTransformations
                .asymptoticTransform(x -> w * activator.activator(x) * this.getProbabilityDensity(x), t);
        return Trapezoidal.trapz(new DoubleFunction(integrand), -1, 1, TRAP_COUNT);
    }

    /**
     * Get the variance of a distribution whose underlying data has undergone the
     * transformation of the activator using the transformed mean
     * 
     * @param activator
     * @return
     */
    public double getVarianceOfAppliedActivation(ActivationFunction activator, double w, double mean) {
        DoubleUnaryOperator integrand = t -> IntegralTransformations
                .asymptoticTransform(
                        x -> Math.pow(w * activator.activator(x) - mean, 2) * this.getProbabilityDensity(x), t);
        return Math.sqrt(Trapezoidal.trapz(new DoubleFunction(integrand), -1, 1, TRAP_COUNT));
    }

    /**
     * Get the variance of a distribution whose underlying data has undergone the
     * transformation of the activator
     * 
     * @param activator
     * @return
     */
    public double getVarianceOfAppliedActivation(ActivationFunction activator, double w) {
        return getVarianceOfAppliedActivation(activator, w, getMeanOfAppliedActivation(activator, w));
    }

    public DoubleUnaryOperator getActivatedDistribution(ActivationFunction activator, double weight) {
        
        throw new RuntimeException("This method has been depricated"); 
        // return x -> getProbabilityDensity(activator.inverse(x) / weight)
        //         * Math.abs(activator.inverseDerivative(x) / weight);
    }

    public IConvolution toConvolution(ActivationFunction activator, double weight){
        DoubleUnaryOperator activatedDist = getActivatedDistribution(activator, weight);
        return new GenericConvolution(activatedDist);
    }
}
