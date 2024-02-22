package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.NelderMead;

public class BellCurveDistributionAdjuster implements Function {

    private static final double TOLLERANCE = 1E-6; // tollerance for optimization
    private static final double DIGITS_OF_PRECISION = 10;
    private static final int integrationDivisions = 10000;
    private static final int NM_ITTERATION_LIMIT = 10000;

    private static final double root_pi = Math.sqrt(Math.PI);
    private static final double root_2 = Math.sqrt(2d);
    private static final double root_2pi = root_pi * root_2;

    // range of values to pre-compute for the relative shift from [-w_domain,
    // w_domain]
    private static final double w_domain = 10;

    // number of pre-computed relative shift values
    private static final int w_divisions = 500;

    // range of values to pre-compute for the relative scale from [1/eta_domain,
    // eta_domain]
    private static final double eta_domain = 10;

    // number of pre-computed relative scale values
    private static final int eta_divisions = 500;

    /**
     * Expected likelihood value map dimensions are computed as [w][eta]
     */
    private static LinearInterpolation2D expectationMap;

    // Indicates whether the expectation map has been initialized
    private static boolean is_map_initialized = false;

    // The distribution that is being updated
    private final BellCurveDistribution parentDistribution;

    // The mean value of the parent distribution
    private double mean;

    // The variance of the parent distribution
    private double variance;

    // The number of points in the parent distribution
    private double N;

    // All distributions which will create a residual in updating the parent
    // distribution
    private ArrayList<BellCurveDistribution> influincingDistributions;

    // The weight of each distribution
    private ArrayList<Double> distribution_weights;

    // The x-values of all points being used to reinforce/diminish the distribution
    private ArrayList<Double> update_points;

    // The reinforcement value for each point. I.E b = true for reinforcement, b =
    // false for diminishment
    private ArrayList<Boolean> points_b;

    // The weight of each point
    private ArrayList<Double> point_weights;

    private final boolean is_using_map;

    public BellCurveDistributionAdjuster(BellCurveDistribution parentDistribution, boolean use_map) {
        this.parentDistribution = parentDistribution;
        is_using_map = use_map;
        mean = parentDistribution.getMean();
        variance = parentDistribution.getVariance();
        N = parentDistribution.getN();

        influincingDistributions = new ArrayList<>();
        distribution_weights = new ArrayList<>();
        update_points = new ArrayList<>();
        points_b = new ArrayList<>();
        point_weights = new ArrayList<>();

        if (use_map & !is_map_initialized) {
            Range w_range = new LinearRange(-w_domain, w_domain, w_divisions, true, true);
            Range eta_range = new MultiplicitiveRange(1 / eta_domain, eta_domain, eta_divisions, true, true);

            expectationMap = new LinearInterpolation2D(w_range, eta_range,
                    BellCurveDistributionAdjuster::computeNonAnalyticComponent, true);
        }
    }

    public BellCurveDistributionAdjuster(BellCurveDistribution parentDistribution) {
        this(parentDistribution, true);
    }

    // Set up functions

    /**
     * Reinforce or diminish this distribution for a given point
     * 
     * @param x
     * @param isReinforcing
     * @param weight
     */
    public void addPoint(double x, boolean isReinforcing, double weight) {
        update_points.add(x);
        points_b.add(isReinforcing);
        point_weights.add(weight);
    }

    /**
     * Reinforce or diminish this distribution for a given distribution
     * 
     * @param bcd
     * @param weight
     */
    public void addDistribution(BellCurveDistribution bcd, double weight) {
        influincingDistributions.add(bcd);
        distribution_weights.add(weight);
    }

    // End set up

    private void clear() {
        influincingDistributions.clear();
        distribution_weights.clear();
        update_points.clear();
        points_b.clear();
        point_weights.clear();
    }

    public double getMean() {
        return mean;
    }

    public double getVariance() {
        return variance;
    }

    public double getN() {
        return N;
    }

    public void applyAdjustments() {
        NelderMead nm = new NelderMead();
        List<Vec> init_points = new ArrayList<>(3);

        init_points.add(new DenseVector(new double[] { -0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0, 0.2 }));

        Vec solution = nm.optimize(TOLLERANCE, NM_ITTERATION_LIMIT, this, init_points);
        mean += solution.get(0);
        variance *= Math.exp(solution.get(1));

        double weight_sum = point_weights.stream().mapToDouble(w -> w).sum();
        weight_sum += distribution_weights.stream().mapToDouble(w -> w).sum();
        N += weight_sum;
        clear();
    }

    @Override
    public double f(double... theta) {
        return -logLikelihoodOfParameters(theta[0], Math.exp(theta[1])); // use exponential transformation to enforce
                                                                         // that variance > 0
    }

    @Override
    public double f(Vec theta) {
        return f(theta.arrayCopy());
    }

    /**
     * The total log-likelihood for a choice of parameters
     * 
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfParameters(double shift, double scale) {
        double likelihood = parentDistribution.getN() * logLikelihoodOfDistribution(parentDistribution, shift, scale);
        likelihood += logLikelihoodOfPoints(shift, scale);
        likelihood += logLikelihoodOfDistributions(shift, scale);

        return likelihood;
    }

    /**
     * Get the cumulative log-likelihood of all points added to this adjuster
     * 
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfPoints(double shift, double scale) {
        final double mean = this.mean + shift;
        final double variance = this.variance * scale;
        return IntStream.range(0, update_points.size())
                .mapToDouble(i -> point_weights.get(i)
                        * logLikelihood(update_points.get(i), points_b.get(i), mean, variance))
                .sum();
    }

    /**
     * The log-likelihood for a single data point with position x and reinforcement
     * value b
     * 
     * @param x        position
     * @param b        reinforcement state (true for reinforcment, false for
     *                 diminishment)
     * @param mean
     * @param variance
     * @return
     */
    public double logLikelihood(double x, boolean b, double mean, double variance) {
        if (x == mean)
            return 0;

        final double rate = -(x - mean) * (x - mean) / (2 * variance * variance);
        final double rate_exp = Math.exp(rate);
        final double omega = -Math.log(2 * Math.PI * variance * variance) / 2;
        if (b) {
            return 2 * rate + omega;
        } else {
            return Math.log(1 - rate_exp) + rate + omega;
        }
    }

    /**
     * The cumulative log-likelihood of all distributions
     * 
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfDistributions(double shift, double scale) {
        return IntStream.range(0, influincingDistributions.size())
                .mapToDouble(i -> distribution_weights.get(i)
                        * logLikelihoodOfDistribution(influincingDistributions.get(i), shift, scale))
                .sum();
    }

    /**
     * The log-likelihood of a distribution for a set of shift and scale parameters
     * 
     * @param bcd
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfDistribution(BellCurveDistribution bcd, double shift, double scale) {
        final double w = getRelativeShift(bcd, shift, scale);
        final double eta = getRelativeScale(bcd, shift, scale);

        double analytic_comp = getAnalyticComponent(scale, w, eta);
        double non_analytic_comp;
        if (is_using_map) {
            non_analytic_comp = getNonAnalyticComponent(w, eta);
        } else {
            non_analytic_comp = computeNonAnalyticComponent(w, eta);
        }
        return analytic_comp + non_analytic_comp;
    }

    private double getAnalyticComponent(double scale, double w, double eta) {
        double value = (root_2 + 1) * w * w;
        value += (2 * root_2 + 1) / (4 * eta);
        value /= root_2;

        value += Math.log(2 * Math.PI * scale * scale * variance * variance) / 2;
        return -value;
    }

    private static double getNonAnalyticComponent(double w, double eta) {
        return expectationMap.interpolate(w, eta);
    }

    private static double computeNonAnalyticComponent(double w, double eta) {
        return Math.sqrt(eta) * infiniteIntegral(x -> logLikelihoodProbabilityDensity(x, w, eta)) / root_pi;
    }

    /**
     * Probability density function with parameters w and eta
     * 
     * @param x
     * @param w
     * @param eta
     * @return
     */
    public static double logLikelihoodProbabilityDensity(double x, double w, double eta) {
        final double rate_exp_plus = Math.exp(-eta * (x + w) * (x + w));
        final double rate_exp_minus = Math.exp(-eta * (x - w) * (x - w));
        final double erf = Math.exp(-x * x);

        double expectation_function = Math.log(1 - erf);
        double p_dist_plus = (1 - rate_exp_plus) * rate_exp_plus;
        double p_dist_minus = (1 - rate_exp_minus) * rate_exp_minus;

        return expectation_function * (p_dist_plus + p_dist_minus);
    }

    /**
     * compute the relative shift parameter "w"
     * 
     * @param bcd
     * @return
     */
    private double getRelativeShift(BellCurveDistribution bcd, double shift, double scale) {
        return (bcd.getMean() - mean - shift) / (root_2 * scale * variance);
    }

    /**
     * Compute the relative scale parameter "eta"
     * 
     * @param bcd
     * @return
     */
    private double getRelativeScale(BellCurveDistribution bcd, double shift, double scale) {
        double eta = scale * variance / bcd.getVariance();
        return eta * eta;
    }

    /**
     * Integrate a function on the bounds of [a, b]
     * 
     * @param func
     * @param a
     * @param b
     * @return
     */
    public static double integrate(DoubleUnaryOperator func, double a, double b) {
        Range t_range = new LinearRange(a, b, integrationDivisions - 2, false, false);
        double intermediate = t_range.stream().map(func).sum();
        intermediate += (func.applyAsDouble(a) + func.applyAsDouble(b)) / 2;
        return intermediate * (b - a) / integrationDivisions;
    }

    /**
     * Integrates the given function on the bounds [0, infinity) 
     * 
     * @return
     */
    public static double infiniteIntegral(DoubleUnaryOperator func) {
        double intVal = integrate((double t) -> IntegralTransformations.hurwitzThreeHalfsTransform(func, t), 0, 1);
        assert Double.isFinite(intVal);
        return intVal;
    }

}
