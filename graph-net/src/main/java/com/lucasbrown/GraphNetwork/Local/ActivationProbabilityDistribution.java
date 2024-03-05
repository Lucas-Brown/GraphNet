package com.lucasbrown.GraphNetwork.Local;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.DoubleStream;

import com.lucasbrown.NetworkTraining.ApproximationTools.DoubleFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.IntegralTransformations;

import jsat.math.integration.Romberg;
import jsat.math.integration.Trapezoidal;

/**
 * Contains the probability distribution information for likelyhood of a signal
 * being sent from one node to another.
 * 
 * TODO: alter methods to include negative reinforcement as well
 */
public abstract class ActivationProbabilityDistribution {
    
    private static final int TRAP_COUNT = 100;

    /**
     * Random number generator for probabalistically choosing whether to send a
     * signal
     */
    protected Random rand = new Random();

    /**
     * Returns the probability density at a point x
     * 
     * @param x
     * @return
     */
    public abstract double getProbabilityDensity(double x);

    public abstract double sample();

    public abstract boolean shouldSend(double inputSignal);

    public abstract void prepareReinforcement(double valueToReinforce);

    public abstract void prepareDiminishment(double valueToDiminish);

    public abstract double getMean();

    public abstract double getVariance();

    /**
     * Apply adjustments from reinforcing/diminishing the distribution
     */
    public abstract void applyAdjustments();

    /**
     * Difference between the most likely outcome and the given outcome x
     * 
     * @param x
     * @return
     */
    public abstract double differenceOfExpectation(double x);

    
    public double[] sample(int count)
    {
        return DoubleStream.generate(this::sample).limit(count).toArray();
    }


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
                .asymptoticTransform(x -> w*activator.activator(x) * this.getProbabilityDensity(x), t);
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
                .asymptoticTransform(x -> Math.pow(w*activator.activator(x) - mean, 2) * this.getProbabilityDensity(x), t);
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
}
