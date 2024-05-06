package com.lucasbrown.GraphNetwork.Distributions;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.IConvolution;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.LinearBellConvolution;
import com.lucasbrown.NetworkTraining.DataSetTraining.BellCurveDistributionAdjuster;



/**
 * A Bernoulli distribution with p = p(x) = Normal(mean, variance)
 */
public class BellCurveFilter extends Filter{

    /**
     * mean value and standard deviation of a normal distribution
     */
    private double mean, variance;

    /**
     * The number of "data points" that this distribution represents
     */
    private double N;

    
    /**
     * Maximum number of points this distribution can represent.
     */
    private double N_max;

    /**
     * An object to apply adjustments to this distribution given new data
     */
    private BellCurveDistributionAdjuster adjuster;

    public BellCurveFilter(BellCurveFilter toCopy){
        this(toCopy.mean, toCopy.variance, toCopy.N);
    }

    /**
     * @param mean     mean value of the normal distribution
     * @param variance variance of the normal distribution
     */
    public BellCurveFilter(double mean, double variance) {
        this(mean, variance, 10);
    }

    public BellCurveFilter(double mean, double variance, double N) {
        this(mean, variance, N, 1000);
    }

    /**
     * @param mean     mean value of the normal distribution
     * @param variance variance of the normal distribution
     * @param N        the number of points this distribution approximates
     */
    public BellCurveFilter(double mean, double variance, double N, double N_max) {
        this.mean = mean;
        this.variance = variance;
        this.N = N;
        this.N_max = N_max;
        this.adjuster = new BellCurveDistributionAdjuster(this, true);
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
    public void prepareReinforcement(double valueToReinforce, double weight) {
        adjuster.addPoint(valueToReinforce, true, weight);
    }

    /**
     * Diminish the distribution using {@code valueToDiminish}.
     * 
     * @param valueToDiminish The data point to diminish the likelihood
     */
    @Override
    public void prepareDiminishment(double valueToDiminish, double weight) {
        adjuster.addPoint(valueToDiminish, false, weight); 
    }

    @Override 
    public double sendChance(double inputSignal)
    {
        return computeNormalizedDist(inputSignal);
    }

    @Override
    public double getProbabilityDensity(double x) {
        return computeNormalizedDist(x)/(Math.sqrt(Math.PI) * variance);
    }

    @Override
    public double sample() {
        return rand.nextGaussian()*variance/Math.sqrt(2) + mean;
    }

    @Override
    public double getMeanOfAppliedActivation(ActivationFunction activator, double w) {
        if(activator instanceof ActivationFunction.Linear)
        {
            return mean;
        }
        else{
            return super.getMeanOfAppliedActivation(activator, w);
        }
    }

    @Override
    public double getVarianceOfAppliedActivation(ActivationFunction activator, double w, double mean) {
        if(activator instanceof ActivationFunction.Linear)
        {
            return variance;
        }
        else{
            return super.getVarianceOfAppliedActivation(activator, w, mean);
        }
    }

    @Override
    public double differenceOfExpectation(double x) {
        return mean - x; //1 - computeNormalizedDist(x);
    }

    @Override
    public double getMean(){
        return mean;
    }

    @Override
    public double getVariance() {
        return variance/Math.sqrt(2);
    }

    @Override
    public double getNumberOfPointsInDistribution(){
        return N;
    }

    @Override
    public void applyAdjustments() {
        adjuster.applyAdjustments();
        mean = adjuster.getMean();
        variance = adjuster.getVariance();
        N = Math.min(adjuster.getN(), N_max);
    }

    private static double NormalizedDist(double x, double mean, double variance) {
        final double d = (x - mean)/variance;
        return Math.exp(-d * d); // no divide by 2
    }

    @Override
    public Filter copy() {
        return new BellCurveFilter(this);
    }

    @Override
    public IConvolution toConvolution(ActivationFunction activator, double weight) {
        if(activator.equals(ActivationFunction.LINEAR)){
            return new LinearBellConvolution(this, weight);
        }

        return super.toConvolution(activator, weight);
    }

}
