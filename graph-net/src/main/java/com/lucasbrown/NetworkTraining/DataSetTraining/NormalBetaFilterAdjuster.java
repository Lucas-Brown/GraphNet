package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.ArrayList;
import java.util.List;

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
    private static final int TRAPZ_STEPS = 1000; // Number of steps for numerical integration
    private static final int NM_ITTERATION_LIMIT = 1000; // Max iterations for Nelder-Mead optimization

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

    // Range and resolution for precomputed expected likelihood values
    private static final double w_domain = 5; // Range for relative shift 'w'
    private static final int w_divisions = 1000; // Number of divisions for 'w'
    private static final double eta_domain = 5; // Range for relative scale 'eta'
    private static final int eta_divisions = 1000; // Number of divisions for 'eta'

    // Precomputed table of expected likelihood values for different shifts and
    // scales
    private static LinearInterpolation2D expectationMap;

    // Flag to indicate if the expectation map is initialized
    private static boolean is_map_initialized = false;

    // Indicates whether to use the precomputed expectation map
    private final boolean is_using_map;

    public NormalBetaFilterAdjuster(NormalBetaFilter filter, NormalDistribution nodeDistribution,
            BetaDistribution arcDistribution, boolean use_map) {
        this.filter = filter;
        this.nodeDistribution = nodeDistribution;
        this.arcDistribution = arcDistribution;
        is_using_map = use_map;

        adjustementPoints = new ArrayList<>();

        if (is_using_map & !is_map_initialized) {
            Range w_range = new LinearRange(-w_domain, w_domain, w_divisions, true, true);
            Range eta_range = new MultiplicitiveRange(1 / eta_domain, eta_domain, eta_divisions, true, true);

            expectationMap = new LinearInterpolation2D(w_range, eta_range,
                    NormalBetaFilterAdjuster::computeC, true);
            is_map_initialized = true;
        }
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
        prepareAdjustment(weight, newData[0], newData[1]);
    }

    /**
     * Prepares the adjuster to incorporate a new data point with a weight.
     * This stores the data point along with its weight for later use in the
     * adjustment process.
     *
     * @param weight The weight to assign to the new data point.
     * @param x      The x value of the data point.
     * @param b      The beta observation (0 to 1) of the data point.
     */
    public void prepareAdjustment(double weight, double x, double b) {
        adjustementPoints.add(new WeightedPoint<FilterPoint>(weight, new FilterPoint(x, b)));
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
        init_points.add(new DenseVector(new double[] { -0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0, 0.2 }));

        // Optimize the function (negative log-likelihood) to find the best shift and
        // scale
        Vec solution = nm.optimize(TOLLERANCE, NM_ITTERATION_LIMIT, this, init_points);

        // Update the mean and variance based on the optimization result
        mean += solution.get(0); // Adjust the mean
        variance *= scaleTransform(solution.get(1)); // Adjust the variance

        // Update the sample size (N) based on the weights of the new data points
        N += adjustementPoints.stream().mapToDouble(point -> point.weight).sum();
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
        return point.weight * getLogLikelihoodOfPoint(point.value.x, point.value.b, shift, scale);
    }

    /**
     * Calculates the log-likelihood of a single data point given the filter
     * parameters and adjustments.
     * This reflects how well the filter, with the given adjustments, explains this
     * specific observation.
     *
     * @param x     The observed value of the node.
     * @param b     The observed beta value (success or failure: 1 or 0).
     * @param shift The shift applied to the mean of the filter.
     * @param scale The scaling factor applied to the variance of the filter.
     * @return The log-likelihood of observing this data point.
     */
    public double getLogLikelihoodOfPoint(double x, double b, double shift, double scale) {
        // Calculate the full likelihood of observing x given the adjusted filter
        // parameters
        double full_send = NormalBetaFilter.likelihood(x, mean + shift, scale * variance);

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
        // Calculate intermediate values based on the distributions and adjustments
        double variance_x = nodeDistribution.getVariance();
        final double w = (mean - nodeDistribution.getMean() - shift) / (root_2 * variance_x);
        final double root_eta = variance_x / (scale * variance);
        double alpha = arcDistribution.getAlpha();
        double beta = arcDistribution.getBeta();

        // Calculate terms C and M used in the expected log-likelihood formula
        double C = getC(w, root_eta);
        double M = getM(w, root_eta);

        // Return the expected log-likelihood based on C, M, alpha, and beta
        return (M * alpha + C * beta) / (alpha + beta);
    }

    /**
     * Calculates the M term used in the expected log-likelihood calculation.
     *
     * @param w        The relative shift.
     * @param root_eta The square root of the relative scale.
     * @return The value of M.
     */
    public double getM(double w, double root_eta) {
        return root_eta * root_eta * root_pi * (1 + 2 * w * w) / 2;
    }

    /**
     * Gets the pre-computed or calculated C term used in the expected
     * log-likelihood calculation.
     *
     * @param w        The relative shift.
     * @param root_eta The square root of the relative scale.
     * @return The value of C.
     */
    public double getC(double w, double root_eta) {
        if (is_using_map) {
            // Use pre-computed value if available
            return expectationMap.interpolate(w, root_eta);
        } else {
            // Otherwise, compute it on the fly
            return computeC(w, root_eta);
        }
    }

    /**
     * Computes the C term used in the expected log-likelihood calculation.
     * This involves numerical integration (trapezoidal rule) of a function.
     *
     * @param w        The relative shift.
     * @param root_eta The square root of the relative scale.
     * @return The value of C.
     */
    public final static double computeC(double w, double root_eta) {
        // Define the integrand function for the C calculation
        DoubleFunction integrand = new DoubleFunction((double t) -> finiteIntegrand(t, w, root_eta));

        // Perform numerical integration using the trapezoidal rule
        return Trapezoidal.trapz(integrand, -1, 1, TRAPZ_STEPS);
    }

    /**
     * Returns the evaluation of the C-constant integrand.
     * 
     * @param x        The evaluation point
     * @param w        The relative shift
     * @param root_eta the square root of the relative scale
     * @return The integrand evaluation at x
     */
    public static final double CIntegrand(double x, double w, double root_eta) {
        // Collapse the integral from (-inf, inf) to [0, inf) by reflecting the function
        // onto itself.
        double left_shift = x / root_eta + w;
        double right_shift = x / root_eta - w;

        // Likelihood of x
        double left = Math.exp(-left_shift * left_shift);
        double right = Math.exp(-right_shift * right_shift);
        double result = left + right;

        // Expectation weight for x
        // approximation has less than a 0.1% error.
        if (x < 0.1) {
            result *= 2 * Math.log(x);
        } else {
            result *= Math.log(1 - Math.exp(-x * x));
        }
        assert Double.isFinite(result);
        return result;
    }

    /**
     * Transforms the bounds of the integrand from [0, -inf) to [-1, 1] using a
     * specialized transformation x = e^(t/ln|t|)
     * 
     * @param t        The evaluation point between -1 and 1
     * @param w        The relative shift
     * @param root_eta The square root of the relative scale
     * @return the integrand evaluated at x = e^(t/ln|t|)
     */
    public static double finiteIntegrand(double t, double w, double root_eta) {
        return IntegralTransformations.expInvLogTransform(x -> CIntegrand(x, w, root_eta), t);
    }

    /**
     * A point pair object for the x-value and success rate
     */
    private static class FilterPoint {

        // The x-value
        public double x;

        // The success rate
        public double b;

        /**
         * Constructs a filter point from a x value and success rate pair
         * 
         * @param x The x-value
         * @param b The success rate
         */
        public FilterPoint(double x, double b) {
            this.x = x;
            this.b = b;
        }

        @Override
        public String toString() {
            return "value: " + Double.toString(x) + "\tb: " + Double.toString(b);
        }
    }

}
