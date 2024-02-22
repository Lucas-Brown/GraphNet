package com.lucasbrown.GraphNetwork.Local;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;
import java.util.stream.IntStream;

import com.lucasbrown.NetworkTraining.BellCurveDistributionAdjuster;

/**
 * A Bernoulli distribution with p = p(x) = Normal(mean, variance)
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
     * An object to apply adjustments to this distribution given new data
     */
    private BellCurveDistributionAdjuster adjuster;

    /**
     * @param mean     mean value of the normal distribution
     * @param variance variance of the normal distribution
     */
    public BellCurveDistribution(double mean, double variance) {
        this(mean, variance, 10);
    }

    /**
     * @param mean     mean value of the normal distribution
     * @param variance variance of the normal distribution
     * @param N        the number of points this distribution approximates
     */
    public BellCurveDistribution(double mean, double variance, double N) {
        this.mean = mean;
        this.variance = variance;
        this.N = N;
        this.adjuster = new BellCurveDistributionAdjuster(this);
    }

    /**
     * normalized normal distribution
     * 
     * @param x
     * @return value of the distribution at the point x. always returns on the
     *         interval (0, 1]
     */
    public double computeNormalizedDist(double x) {
        return NormalizedDist(x, mean, variance);
    }

    /**
     * Reinforce the mean and standard deviation with {@code valueToReinforce}.
     * 
     * @param valueToReinforce The new data to add to the distribution data set
     */
    @Override
    public void prepareReinforcement(double valueToReinforce) {
        adjuster.addPoint(valueToReinforce, true, 1/getProbabilityDensity(valueToReinforce));
    }

    /**
     * Diminish the distribution using {@code valueToDiminish}.
     * 
     * @param valueToDiminish The data point to diminish the likelihood
     */
    @Override
    public void prepareDiminishment(double valueToDiminish) {
        adjuster.addPoint(valueToDiminish, false, 1/(1-getProbabilityDensity(valueToDiminish)));
    }

    @Override
    public boolean shouldSend(double inputSignal) {
        // Use the normalized normal distribution as a measure of how likely
        return computeNormalizedDist(inputSignal) >= rand.nextDouble();
    }

    @Override
    public double getProbabilityDensity(double x) {
        return computeNormalizedDist(x)/(Math.sqrt(Math.PI) * variance);
    }

    @Override
    public double differenceOfExpectation(double x) {
        return 1 - computeNormalizedDist(x);
    }

    @Override
    public double getMean(){
        return mean;
    }

    @Override
    public double getVariance() {
        return variance;
    }

    public double getN() {
        return N;
    }

    @Override
    public void applyAdjustments() {
        adjuster.applyAdjustments();
        mean = adjuster.getMean();
        variance = adjuster.getVariance();
        N = adjuster.getN();
    }

    private static double NormalizedDist(double x, double mean, double variance) {
        final double d = (x - mean)/variance;
        return Math.exp(-d * d / 2);
    }

}
