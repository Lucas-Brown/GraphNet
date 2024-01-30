package src.GraphNetwork.Local;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import src.GraphNetwork.Global.DoublePair;
import src.NetworkTraining.LinearInterpolation2D;
import src.NetworkTraining.LinearRange;
import src.NetworkTraining.Range;

/**
 * A distribution which is parameterized similarly to a bell curve but functionally distinct.
 * This distribution weighs reinforcment data points higher than diminishment points.
 * 
 * Every point on the real line needs to be characterized by a bernoulli distribution to represent the reinforcment or diminishment of a datum
 * This creates an infinite-dimensional parameter space which must be approximated.
 * This distribution makes the assumption that the success probability follows an un-normalized bell curve.
 * i.e: p(x) = exp(-(x-mean)^2 /(2 variance^2))
 *
 * In order to update the distribution based on new data, it is necessary to use a method which does not rely on past data.
 * Here, I approximate the data set as a sufficient model for the data, thus allowing for finite sums to be approximated as integrals.
 * The resulting integrals are related to the Hurwitz-zeta function and require additional approximations to work with.
 * The shift (change in the mean) and the scale (factor for the variance) need to be solved for itteratively using Newton's method.
 * Since newton's method is rather expensive in context, the equations for both the shift and scale have been factored into a residual and dynamic component.
 * The residual components are expensive but can be pre-computed without knowing the mean or variance that's being solved for.
 * The dynamic component is less expensive but cannot be pre-computed.
 * In all, the resulting approximations tend to underestimate the shift slightly and scale moderately. 
 * 
 * If you're reading this, good luck lol
 */
public class BellCurveDistribution extends ActivationProbabilityDistribution {

    /**
     * mean value and standard deviation of a normal distribution
     */
    private double mean, variance;

    /**
     * The number of "data points" that this distribution represents 
     */
    private double N;

    /**
     * @param mean mean value of the normal distribution
     * @param variance variance of the normal distribution
     */
    public BellCurveDistribution(double mean, double variance)
    {
        this(mean, variance, 10);
    }

    
    /**
     * @param mean mean value of the normal distribution
     * @param variance variance of the normal distribution
     * @param N the number of points this distribution approximates
     */
    public BellCurveDistribution(double mean, double variance, double N)
    {
        this.mean = mean;
        this.variance = variance;
        this.N = N;
    }

    /**
     * normalized normal distribution
     * @param x 
     * @return value of the distribution at the point x. always returns on the interval (0, 1]
     */
    public double computeNormalizedDist(double x)
    {
        return NormalizedDist(x, mean, variance);
    }

    /**
     * Reinforce the mean and standard deviation with {@code valueToReinforce}.
     * @param valueToReinforce The new data to add to the distribution data set
     */
    @Override
    public void reinforceDistribution(double valueToReinforce)
    {   
        newtonUpdateMeanAndVariance(valueToReinforce, true);
    }

    /**
     * reinforces the distribution directly, not accounting for its role within the larger network
     * @param valueToReinforce
     */
    public void reinforceDistributionNoFilter(double valueToReinforce)
    {   
        newtonUpdateMeanAndVariance(valueToReinforce, true, 1);
    }
    
    /**
     * Diminish the distribution using {@code valueToDiminish}.
     * @param valueToDiminish The data point to diminish the likelihood
     */
    @Override
    public void diminishDistribution(double valueToDiminish) 
    {
        newtonUpdateMeanAndVariance(valueToDiminish, false);
    }

    /**
     * diminishes the distribution directly, not accounting for its role within the larger network
     * @param valueToDiminish
     */
    public void diminishDistributionNoFilter(double valueToDiminish) 
    {
        newtonUpdateMeanAndVariance(valueToDiminish, false, 1);
    }

    @Override
    public boolean shouldSend(double inputSignal) {
        // Use the normalized normal distribution as a measure of how likely  
        return computeNormalizedDist(inputSignal) >= rand.nextDouble();
    }

    @Override
    public double getMeanValue()
    {
        return mean;
    }

    public double getVariance()
    {
        return variance;
    }

    protected double getN()
    {
        return N;
    }

    public void setParamsFromAdjuster(BellCurveDistributionAdjuster bcda)
    {
        mean = bcda.getMean();
        variance = bcda.getVariance();
        N = bcda.getN();
    }

    private static double NormalizedDist(double x, double mean, double variance)
    {
        final double d = x-mean;
        return Math.exp(-d*d/variance/2);
    }

    private void newtonUpdateMeanAndVariance(double x, boolean b)
    {
        // if no weight is specified, assume the weight should be 1/P(x, b)
        double weight = computeNormalizedDist(x);
        weight = b ? 1/weight : 1/(1-weight);
        newtonUpdateMeanAndVariance(x, b, weight);
    }



}
