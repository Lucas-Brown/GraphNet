package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import com.lucasbrown.NetworkTraining.ApproximationTools.DoubleFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.IntegralTransformations;
import com.lucasbrown.NetworkTraining.ApproximationTools.LinearInterpolation2D;
import com.lucasbrown.NetworkTraining.ApproximationTools.LinearRange;
import com.lucasbrown.NetworkTraining.ApproximationTools.MultiplicitiveRange;
import com.lucasbrown.NetworkTraining.ApproximationTools.Range;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.integration.Romberg;
import jsat.math.integration.Trapezoidal;
import jsat.math.optimization.NelderMead;

public class NormalBetaFilterAdjuster implements IExpectationAdjuster, Function {

    private static final double TOLLERANCE = 1E-6; // tollerance for optimization
    private static final int TRAPZ_STEPS = 1000; // number of integration points
    private static final int NM_ITTERATION_LIMIT = 1000;

    protected static final double root_pi = Math.sqrt(Math.PI);
    protected static final double root_2 = Math.sqrt(2d);
    protected static final double root_2pi = root_pi * root_2;

    protected final NormalBetaFilter filter;
    protected final NormalDistribution nodeDistribution;
    protected final BetaDistribution arcDistribution;

    protected double mean, variance, N;

    private ArrayList<WeightedPoint<FilterPoint>> adjustementPoints;

    // range of values to pre-compute for the relative shift from [-w_domain,
    // w_domain]
    private static final double w_domain = 5;

    // number of pre-computed relative shift values
    private static final int w_divisions = 1000;

    // range of values to pre-compute for the relative scale from [1/eta_domain,
    // eta_domain]
    private static final double eta_domain = 5;

    // number of pre-computed relative scale values
    private static final int eta_divisions = 1000;

    /**
     * Expected likelihood value map dimensions are computed as [w][eta]
     */
    private static LinearInterpolation2D expectationMap;

    // Indicates whether the expectation map has been initialized
    private static boolean is_map_initialized = false;

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

    @Override
    public void prepareAdjustment(double weight, double[] newData) {
        prepareAdjustment(weight, newData[0], newData[1]);
    }

    public void prepareAdjustment(double weight, double x, double b) {
        adjustementPoints.add(new WeightedPoint<FilterPoint>(weight, new FilterPoint(x, b)));
    }

    @Override
    public void prepareAdjustment(double[] newData) {
        prepareAdjustment(1, newData);
    }

    @Override
    public void applyAdjustments() {
        mean = filter.getMean();
        variance = filter.getVariance();
        N = filter.getN();

        NelderMead nm = new NelderMead();
        List<Vec> init_points = new ArrayList<>(3);

        init_points.add(new DenseVector(new double[] { -0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0.2, -0.2 }));
        init_points.add(new DenseVector(new double[] { 0, 0.2 }));

        Vec solution = nm.optimize(TOLLERANCE, NM_ITTERATION_LIMIT, this, init_points);
        mean += solution.get(0);
        variance *= scaleTransform(solution.get(1));

        N += adjustementPoints.stream().mapToDouble(point -> point.weight).sum();
        adjustementPoints.clear();

        filter.applyAdjustments(this);
    }

    @Override
    public double[] getUpdatedParameters() {
        return new double[] { mean, variance, N };
    }

    private double scaleTransform(double pre_scale) {
        return Math.exp(pre_scale);
    }

    @Override
    public double f(double... x) {
        return -getLogLikelihood(x[0], scaleTransform(x[1])); // maximize by minimizing the negative 
    }

    @Override
    public double f(Vec x) {
        return f(x.arrayCopy());
    }

    public double getLogLikelihood(double shift, double scale) {
        double expected = N * getExpectedValueOfLogLikelihood(shift, scale);
        double sum = getSumOfWeightedPoints(shift, scale);

        return expected + sum;
    }

    public double getSumOfWeightedPoints(double shift, double scale) {
        return adjustementPoints.stream()
                .mapToDouble(point -> getWeightedLogLikelihoodOfPoint(point, shift, scale))
                .sum();
    }

    public double getWeightedLogLikelihoodOfPoint(WeightedPoint<FilterPoint> point, double shift, double scale) {
        return point.weight * getLogLikelihoodOfPoint(point.value.x, point.value.b, shift, scale);
    }

    public double getLogLikelihoodOfPoint(double x, double b, double shift, double scale) {
        double full_send = NormalBetaFilter.likelihood(x, mean + shift, scale * variance);
        if (b == 1) {
            return Math.log(full_send);
        } else if (b == 0) {
            return (1 - b) * Math.log(1 - full_send);
        } else {
            return b * Math.log(full_send) + (1 - b) * Math.log(1 - full_send);
        }
    }

    public double getExpectedValueOfLogLikelihood(double shift, double scale) {
        double variance_x = nodeDistribution.getVariance();
        final double w = (mean - nodeDistribution.getMean() - shift) / (root_2 * variance_x);
        final double root_eta = variance_x / (scale * variance);
        double alpha = arcDistribution.getAlpha();
        double beta = arcDistribution.getBeta();

        double C = getC(w, root_eta);
        double M = getM(w, root_eta);

        return (M * alpha + C * beta) / (alpha + beta);
    }

    public double getM(double w, double root_eta) {
        return root_eta * root_eta * root_pi * (1 + 2 * w * w) / 2;
    }

    public double getC(double w, double root_eta) {
        if (is_using_map) {
            return expectationMap.interpolate(w, root_eta);
        } else {
            return computeC(w, root_eta);
        }
    }

    public final static double computeC(double w, double root_eta) {
        return Trapezoidal.trapz(new DoubleFunction((double t) -> finiteIntegrand(t, w, root_eta)), -1, 1, TRAPZ_STEPS);
    }

    public static final double CIntegrand(double x, double w, double root_eta) {
        double left_shift = x / root_eta + w;
        double right_shift = x / root_eta - w;

        double left = Math.exp(-left_shift * left_shift);
        double right = Math.exp(-right_shift * right_shift);
        double result = left + right;
        // approximation has less than a 0.1% error. 
        if (x < 0.1) {
            result *= 2 * Math.log(x);
        } else {
            result *= Math.log(1 - Math.exp(-x * x));
        }
        assert Double.isFinite(result);
        return result;
    }

    public static double finiteIntegrand(double t, double w, double root_eta) {
        return IntegralTransformations.expInvLogTransform(x -> CIntegrand(x, w, root_eta), t);
    }

    private static class FilterPoint {
        public double x;
        public double b;

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
