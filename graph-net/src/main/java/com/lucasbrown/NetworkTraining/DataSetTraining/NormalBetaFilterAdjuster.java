package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.ArrayList;
import java.util.List;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.NetworkTraining.ApproximationTools.DoubleFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.IntegralTransformations;
import com.lucasbrown.NetworkTraining.ApproximationTools.LinearInterpolation2D;
import com.lucasbrown.NetworkTraining.ApproximationTools.LinearRange;
import com.lucasbrown.NetworkTraining.ApproximationTools.MultiplicitiveRange;
import com.lucasbrown.NetworkTraining.ApproximationTools.Range;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.integration.Trapezoidal;
import jsat.math.optimization.NelderMead;

/**
 * This class adjusts the parameters of a NormalBetaFilter based on new
 * evidence.
 * It uses an expectation-maximization (EM) approach to find the best-fitting
 * parameters.
 * 
 * A NormalBetaFilter models the probability of a node's true value given
 * observations.
 * - Node values follow a normal distribution.
 * - Observation success (true/false) follows a beta distribution.
 *
 * The EM algorithm iteratively refines the filter's mean and variance to
 * maximize
 * the likelihood of the observed data.
 */
public class NormalBetaFilterAdjuster implements IExpectationAdjuster, Function {

    // Constants for optimization and numerical accuracy
    private static final double TOLLERANCE = 1E-6; // Tolerance for convergence in optimization
    private static final int NM_ITTERATION_LIMIT = 1000; // Max iterations for Nelder-Mead optimization
    private static double ACCURACY = 1E-12;

    // Precomputed mathematical constants for efficiency
    protected static final double root_pi = Math.sqrt(Math.PI);
    protected static final double root_2 = Math.sqrt(2d);
    protected static final double root_2pi = root_pi * root_2;

    // Filter components and adjustable parameters
    protected final NormalBetaFilter filter; // The filter being adjusted
    protected final NormalDistribution nodeDistribution; // Distribution of node values
    protected final BetaDistribution arcDistribution; // Distribution of observation success
    protected double mean, variance, N; // Current mean, variance, and sample size of the filter

    // Data points to be used for adjustment
    private ArrayList<WeightedPoint<FilterPoint>> adjustementPoints;

    public NormalBetaFilterAdjuster(IFilter filter, ITrainableDistribution nodeDistribution,
            ITrainableDistribution arcDistribution) {
        this((NormalBetaFilter) filter, (NormalDistribution) nodeDistribution, (BetaDistribution) arcDistribution);
    }

    public NormalBetaFilterAdjuster(NormalBetaFilter filter, NormalDistribution nodeDistribution,
            BetaDistribution arcDistribution) {
        this.filter = filter;
        this.nodeDistribution = nodeDistribution;
        this.arcDistribution = arcDistribution;

        adjustementPoints = new ArrayList<>();
    }

    /**
     * Prepares the adjuster to incorporate new data point(s) with a weight.
     * This stores the data points along with their weights for later use in the
     * adjustment process.
     *
     * @param weight  The weight to assign to the new data.
     * @param newData The new data, containing the x value and the beta observation
     *                (0 to 1).
     */
    @Override
    public void prepareAdjustment(double weight, double[] newData) {
        prepareAdjustment(weight, newData[0], newData[1], newData[2]);
    }

    /**
     * Prepares the adjuster to incorporate a new data point with a weight.
     * This stores the data point along with its weight for later use in the
     * adjustment process.
     *
     * @param weight The weight to assign to the new data point.
     * @param x      The x value of the data point.
     * @param b      The beta observation (0 to 1) of the data point.
     * @param prob   The probability density of this point being selected
     */
    public void prepareAdjustment(double weight, double x, double b, double prob) {
        adjustementPoints.add(new WeightedPoint<FilterPoint>(weight, new FilterPoint(x, b, prob)));
    }

    /**
     * Prepares the adjuster to incorporate a new data point with default weight
     * (1.0).
     *
     * @param newData The new data, containing the x value and the beta observation
     *                (0 to 1).
     */
    @Override
    public void prepareAdjustment(double[] newData) {
        prepareAdjustment(1, newData);
    }

    /**
     * Applies the accumulated adjustments to the filter parameters.
     * This uses the Nelder-Mead optimization algorithm to find the optimal shift
     * and scale
     * that maximize the log-likelihood of the observed data given the filter model.
     */
    @Override
    public void applyAdjustments() {
        // Retrieve the current filter parameters
        mean = filter.getMean();
        variance = filter.getVariance();
        N = filter.getN();

        // Set up the Nelder-Mead optimization
        NelderMead nm = new NelderMead();
        List<Vec> init_points = new ArrayList<>(3); // Initial points for the optimization

        // Define three initial points around (0, 0)
        init_points.add(new DenseVector(new double[] { -0.5, -0.5 }));
        init_points.add(new DenseVector(new double[] { 0.5, -0.5 }));
        init_points.add(new DenseVector(new double[] { 0, 0.5 }));

        // Optimize the function (negative log-likelihood) to find the best shift and
        // scale
        Vec solution = nm.optimize(TOLLERANCE, NM_ITTERATION_LIMIT, this, init_points);

        // Update the mean and variance based on the optimization result
        mean += solution.get(0); // Adjust the mean
        variance *= scaleTransform(solution.get(1)); // Adjust the variance

        // Update the sample size (N) based on the weights of the new data points
        N += adjustementPoints.stream().mapToDouble(point -> point.weight).sum();
        N = Math.min(N, GraphNetwork.N_MAX);
        adjustementPoints.clear(); // Clear the accumulated data points

        // Apply the adjustments to the filter
        filter.applyAdjustments(this);
    }

    /**
     * Returns the updated parameters of the filter after adjustments.
     *
     * @return An array containing the updated mean, variance, and sample size (N).
     */
    @Override
    public double[] getUpdatedParameters() {
        return new double[] { mean, variance, N };
    }

    /**
     * Transforms a pre-scale value back to the original scale.
     * The pre-scale value is typically used during optimization and is in log
     * space.
     * This method converts it back to the actual scale by exponentiating it.
     *
     * @param pre_scale The pre-scale value in log space.
     * @return The corresponding scale value in the original space.
     */
    private double scaleTransform(double pre_scale) {
        return Math.exp(pre_scale);
    }

    /**
     * Calculates the negative log-likelihood of the observed data given a shift and
     * scale.
     * This is the objective function that is minimized during optimization.
     *
     * @param x An array containing the shift and pre-scale (log-scale) values.
     * @return The negative log-likelihood.
     */
    @Override
    public double f(double... x) {
        return -getLogLikelihood(x[0], scaleTransform(x[1]));
    }

    /**
     * Calculates the negative log-likelihood using a vector input.
     *
     * @param x A vector containing the shift and pre-scale (log-scale) values.
     * @return The negative log-likelihood.
     */
    @Override
    public double f(Vec x) {
        return f(x.arrayCopy());
    }

    /**
     * Calculates the log-likelihood of the observed data given a shift and scale.
     * This combines the expected log-likelihood based on the filter model and the
     * log-likelihood of the individual data points.
     *
     * @param shift The shift applied to the mean of the filter.
     * @param scale The scaling factor applied to the variance of the filter.
     * @return The log-likelihood.
     */
    public double getLogLikelihood(double shift, double scale) {
        double expected = N * getExpectedValueOfLogLikelihood(shift, scale);
        double sum = getSumOfWeightedPoints(shift, scale);
        return expected + sum;
    }

    /**
     * Calculates the sum of the weighted log-likelihoods of all individual data
     * points.
     *
     * @param shift The shift applied to the mean of the filter.
     * @param scale The scaling factor applied to the variance of the filter.
     * @return The sum of the weighted log-likelihoods.
     */
    public double getSumOfWeightedPoints(double shift, double scale) {
        return adjustementPoints.stream()
                .mapToDouble(point -> getWeightedLogLikelihoodOfPoint(point, shift, scale))
                .sum();
    }

    /**
     * Calculates the weighted log-likelihood of a single data point given a shift
     * and scale.
     *
     * @param point The data point, including its x value, beta observation, and
     *              weight.
     * @param shift The shift applied to the mean of the filter.
     * @param scale The scaling factor applied to the variance of the filter.
     * @return The weighted log-likelihood of the data point.
     */
    public double getWeightedLogLikelihoodOfPoint(WeightedPoint<FilterPoint> point, double shift, double scale) {
        FilterPoint fp = point.value;
        return point.weight * getLogLikelihoodOfPoint(fp.x, fp.b, fp.prob, shift, scale);
    }

    /**
     * Calculates the log-likelihood of a single data point given the filter
     * parameters and adjustments.
     * This reflects how well the filter, with the given adjustments, explains this
     * specific observation.
     *
     * @param x     The observed value of the node.
     * @param b     The observed beta value (success or failure: 1 to 0).
     * @param prob  The probability density.
     * @param shift The shift applied to the mean of the filter.
     * @param scale The scaling factor applied to the variance of the filter.
     * @return The log-likelihood of observing this data point.
     */
    public double getLogLikelihoodOfPoint(double x, double b, double prob, double shift, double scale) {
        // Calculate the full likelihood of observing x given the adjusted filter
        // parameters
        double full_send = prob*NormalBetaFilter.likelihood(x, mean + shift, scale * variance);

        // Handle different cases of the beta observation (b):
        if (b == 1) {
            // Success: log-likelihood of the full probability
            return Math.log(full_send);
        } else if (b == 0) {
            // Failure: log-likelihood of the complementary probability (1 - full_send)
            return (1 - b) * Math.log(1 - full_send);
        } else {
            // Intermediate values of b: weighted combination of log-likelihoods
            return b * Math.log(full_send) + (1 - b) * Math.log(1 - full_send);
        }
    }

    /**
     * Calculates the expected value of the log-likelihood over all possible data
     * points.
     * This represents the overall likelihood of the filter model with adjustments,
     * averaged over the expected data distribution.
     *
     * @param shift The shift applied to the mean of the filter.
     * @param scale The scaling factor applied to the variance of the filter.
     * @return The expected value of the log-likelihood.
     */
    public double getExpectedValueOfLogLikelihood(double shift, double scale) {
        double variance_x = nodeDistribution.getVariance();
        final double w = (mean - nodeDistribution.getMean() + shift) / (root_2 * variance_x);
        double eta = variance_x / (scale * variance);
        eta *= eta;
        double alpha = arcDistribution.getAlpha();
        double beta = arcDistribution.getBeta();

        double A = getA(w, eta, variance_x);
        double B = getB(w, eta, variance_x);

        return (A * alpha + B * beta) / (alpha + beta);
    }

    /**
     * Solution to the integral:
     * \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty}\ln\left(\frac{1}{\sqrt{2\pi\sigma_{x}^{2}}}e^{-x^{2}}e^{-\eta\left(x-w\right)^{2}}\right)e^{-x^{2}}dx
     * 
     * @param w
     * @param eta
     * @param sigma_x
     * @param mu
     * @param mu_x
     * @param sigma
     * @return
     */
    public double getA(double w, double eta, double sigma_x) {
        return -(1 + eta + 2 * w * w * eta + Math.log(2 * Math.PI * sigma_x * sigma_x)) / 2;
    }

    /**
     * Compute the integral:
     * \frac{1}{\sqrt{\pi}}\int_{-\infty}^{\infty}\ln\left(1-\frac{1}{\sqrt{2\pi\sigma_{x}^{2}}}e^{-x^{2}}e^{-\eta\left(x-w\right)^{2}}\right)e^{-x^{2}}dx
     * Using a finite series approximation: -\sum_{n=1}^{N}\frac{e^{-\frac{\left(n+1\right)n\eta w^{2}}{1+n+n\eta}}}{n\sqrt{1+n+n\eta}\left(2\pi\sigma_{x}^{2}\right)^{\frac{n}{2}}}
     */
    public double getB(double w, double eta, double sigma_x) {
        double w2 = w * w;
        double prod_term = root_2pi*sigma_x;
        double prod_pow = 1;

        double sum = 0;
        double delta;
        int n = 1;
        do {
            prod_pow *= prod_term;
            double n_eta = n*eta;
            double denom = 1+n+n_eta;
            delta = Math.exp(-(n+1)*n_eta*w2/denom);
            delta /= Math.sqrt(denom);
            delta /= prod_pow;
            sum += delta;
            assert Double.isFinite(delta);
            assert n < 100000;
        } while (delta > ACCURACY);

        return -sum;
    }

    /**
     * A point pair object for the x-value and success rate
     */
    private static class FilterPoint {

        // The x-value
        public double x;

        // The success rate
        public double b;

        // The probability density that this point was generated
        public double prob;

        /**
         * Constructs a filter point from a x value and success rate pair
         * 
         * @param x The x-value
         * @param b The success rate
         * @param b The probability density
         */
        public FilterPoint(double x, double b, double prob) {
            this.x = x;
            this.b = b;
            this.prob = prob;
        }

        @Override
        public String toString() {
            return "value: " + Double.toString(x) + "\tb: " + Double.toString(b) + "\tprobability: "
                    + Double.toString(prob);
        }
    }

}
