package com.lucasbrown.GraphNetwork.Local;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import com.lucasbrown.NetworkTraining.LinearInterpolation2D;
import com.lucasbrown.NetworkTraining.LinearRange;
import com.lucasbrown.NetworkTraining.MultiplicitiveRange;
import com.lucasbrown.NetworkTraining.Range;

import jsat.linear.DenseVector;
import jsat.linear.Vec;
import jsat.math.Function;
import jsat.math.optimization.NelderMead;


public class BellCurveDistributionAdjuster2 implements Function{
    
    private static final double TOLLERANCE = 1E-6; // tollerance for optimization
    private static final double DIGITS_OF_PRECISION = 10;
    private static final int integrationDivisions = 1000;
    private static final int NM_ITTERATION_LIMIT = 10000;

    private static final double root_pi = Math.sqrt(Math.PI);
    private static final double root_2 = Math.sqrt(2d);
    private static final double root_2pi = root_pi*root_2;

    // The distribution that is being updated
    private final BellCurveDistribution parentDistribution;

    // The mean value of the parent distribution
    private double mean;

    // The variance of the parent distribution
    private double variance;

    // The number of points in the parent distribution
    private double N;

    // All distributions which will create a residual in updating the parent distribution
    private ArrayList<BellCurveDistribution> influincingDistributions;

    // The weight of each distribution
    private ArrayList<Double> distribution_weights;

    // The x-values of all points being used to reinforce/diminish the distribution 
    private ArrayList<Double> update_points;

    // The reinforcement value for each point. I.E b = true for reinforcement, b = false for diminishment
    private ArrayList<Boolean> points_b;

    // The weight of each point
    private ArrayList<Double> point_weights;


public BellCurveDistributionAdjuster2(BellCurveDistribution parentDistribution)
{
    this.parentDistribution = parentDistribution;
    mean = parentDistribution.getMeanValue();
    variance = parentDistribution.getVariance();   
    N = parentDistribution.getN();
    
    influincingDistributions = new ArrayList<>();
    distribution_weights = new ArrayList<>();
    update_points = new ArrayList<>();
    points_b = new ArrayList<>();
    point_weights = new ArrayList<>();
}

// Set up functions

/**
 * Reinforce or diminish this distribution for a given point
 * @param x
 * @param isReinforcing
 * @param weight
 */
public void addPoint(double x, boolean isReinforcing, double weight)
{
    update_points.add(x);
    points_b.add(isReinforcing);
    point_weights.add(weight);
}

/**
 * Reinforce or diminish this distribution for a given distribution
 * @param bcd 
 * @param weight
 */
public void addDistribution(BellCurveDistribution bcd, double weight)
{
    influincingDistributions.add(bcd);
    distribution_weights.add(weight);
}

// End set up


    private void clear()
    {
        influincingDistributions.clear();
        distribution_weights.clear();
        update_points.clear();
        points_b.clear();
        point_weights.clear();
    }

    public double getMean()
    {
        return mean;
    }

    public double getVariance()
    {
        return variance;
    }

    public double getN()
    {
        return N;
    }

    public void applyAdjustments()
    {
        NelderMead nm = new NelderMead();
        List<Vec> init_points = new ArrayList<>(3);

        init_points.add(new DenseVector(new double[]{-0.2, -0.2})); 
        init_points.add(new DenseVector(new double[]{0.2, -0.2}));
        init_points.add(new DenseVector(new double[]{0, 0.2}));

        Vec solution = nm.optimize(TOLLERANCE, NM_ITTERATION_LIMIT, this, init_points);
        mean += solution.get(0);
        variance *= Math.log(solution.get(1)); 

        double weight_sum = point_weights.stream().mapToDouble(w -> w).sum();
        weight_sum += distribution_weights.stream().mapToDouble(w -> w).sum();
        N += weight_sum;
        clear();
    }

    @Override
    public double f(double... theta) {
        return -logLikelihoodOfParameters(theta[0], Math.exp(theta[1])); // use exponential transformation to enforce that variance > 0
    }

    @Override
    public double f(Vec theta) {
        return f(theta.arrayCopy());
    }

    /**
     * The total log-likelihood for a choice of parameters
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfParameters(double shift, double scale)
    {
        double likelihood = parentDistribution.getN() * logLikelihoodOfDistribution(parentDistribution, shift, scale);
        likelihood += logLikelihoodOfPoints(shift, scale);
        likelihood += logLikelihoodOfDistributions(shift, scale);

        return likelihood;
    }

    /**
     * Get the cumulative log-likelihood of all points added to this adjuster 
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfPoints(double shift, double scale)
    {
        final double mean = parentDistribution.getMeanValue() + shift;
        final double variance = parentDistribution.getVariance()*scale;
        return IntStream.range(0, update_points.size())
            .mapToDouble(i -> point_weights.get(i) * logLikelihood(update_points.get(i), points_b.get(i), mean, variance))
            .sum();
    }

    /**
     * The log-likelihood for a single data point with position x and reinforcement value b
     * @param x position
     * @param b reinforcement state (true for reinforcment, false for diminishment)
     * @param mean 
     * @param variance 
     * @return
     */
    public double logLikelihood(double x, boolean b, double mean, double variance)
    {
        final double rate = -(x-mean)*(x-mean)/ (2 * variance * variance);
        final double rate_exp = Math.exp(rate);
        if(b)
        {
            return rate/rate_exp;
        }
        else
        {
            return Math.log(1-rate_exp)/(1-rate_exp);
        }
    }

    /**
     * The cumulative log-likelihood of all distributions
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfDistributions(double shift, double scale)
    {
        return IntStream.range(0, influincingDistributions.size())
            .mapToDouble(i -> distribution_weights.get(i) * logLikelihoodOfDistribution(influincingDistributions.get(i), shift, scale))
            .sum();
    }

    /**
     * The log-likelihood of a distribution for a set of shift and scale parameters
     * @param bcd
     * @param shift
     * @param scale
     * @return
     */
    public double logLikelihoodOfDistribution(BellCurveDistribution bcd, double shift, double scale)
    {
        final double w = getRelativeShift(bcd, shift, scale);
        final double eta = getRelativeScale(bcd, shift, scale);

        double omega = 2 * infiniteIntegral(this::likelihoodProbabilityVolumeDensity);
        return Math.sqrt(eta) * infiniteIntegral(x -> logLikelihoodProbabilityDensity(x, w, eta)) / omega;
    }

    public double likelihoodProbabilityVolumeDensity(double x)
    {
        final double erf = Math.exp(-x*x);
        return likelihoodHelper(erf) * erf;
    }

    /**
     * Probability density function with parameters w and eta
     * @param x
     * @param w
     * @param eta
     * @return
     */
    public double logLikelihoodProbabilityDensity(double x, double w, double eta)
    {
        // As x becomes large, the probability density goes to 0
        /* 
        double w_abs = Math.abs(w);
        boolean cond1 = x >= w_abs + Math.sqrt(DIGITS_OF_PRECISION / (Math.log10(Math.E)) * eta);
        boolean cond2 = x >= Math.sqrt(-Math.log(1-Math.exp(Math.pow(10, -DIGITS_OF_PRECISION))));
        if(cond1 && cond2)
        {
            return 0; 
        }

        double value = 2 - Math.exp(-eta * (w-x)*(w-x)) - Math.exp(-eta * (w+x)*(w+x));
        return value * Math.log(-Math.expm1(-x*x));
        */

        final double rate_exp_plus = Math.exp(-eta*(x+w)*(x+w));
        final double rate_exp_minus = Math.exp(-eta*(x-w)*(x-w));
        final double erf = Math.exp(-x*x);

        double expectation_function = likelihoodHelper(erf);
        expectation_function = Math.log(expectation_function)/expectation_function;

        double p_dist_plus = likelihoodHelper(rate_exp_plus)*rate_exp_plus;
        double p_dist_minus = likelihoodHelper(rate_exp_minus)*rate_exp_minus;

        return expectation_function * (p_dist_plus + p_dist_minus);
    }

    /**
     * compute the relative shift parameter "w"
     * @param bcd
     * @return
     */
    private double getRelativeShift(BellCurveDistribution bcd, double shift, double scale)
    {
        return (bcd.getMeanValue() - mean - shift) / (root_2 * scale * variance);
    }

    /**
     * Compute the relative scale parameter "eta"
     * @param bcd
     * @return
     */
    private double getRelativeScale(BellCurveDistribution bcd, double shift, double scale)
    {
        double eta = scale * variance / bcd.getVariance();
        return eta * eta;
    }

    public static double likelihoodHelper(double exp)
    {
        return Math.pow(exp, exp) * Math.pow(1 - exp, 1 - exp);
    }

    /**
     * Use the transformation x = (1/t - 1)^(3/2) to convert an integral from the bounds [0, Infinity) to [0, 1]
     * @param func the function being integrated over. 
     * @param t the evaluation point on the bounds [0, 1]
     * @return the transformed value at the given point
     */
    public static double infiniteToFiniteIntegralTransform(DoubleUnaryOperator func, double t)
    {
        final double temp = 1/t - 1;

        // if x is effectively infinite, then the provided function is assumed to have a value of 0 due to implicit convergence requirement
        if(Double.isInfinite(temp) || temp == 0)
        {
            return 0;
        }
        
        final double transformedIntegral = func.applyAsDouble(Math.pow(temp, 3d/2)) * Math.sqrt(temp) / (t * t);
        
        if(!Double.isFinite(transformedIntegral))
        {
            System.out.println();
        }
        assert Double.isFinite(transformedIntegral);
        return 3*transformedIntegral/2;
    }

    /**
     * Integrate a function on the bounds of [a, b]
     * @param func
     * @param a
     * @param b
     * @return
     */
    public static double integrate(DoubleUnaryOperator func, double a, double b)
    {
        Range t_range = new LinearRange(a, b, integrationDivisions-2, false, false);
        double intermediate = t_range.stream().map(func).sum();
        intermediate += (func.applyAsDouble(a) + func.applyAsDouble(b)) / 2;
        return intermediate*(b-a)/integrationDivisions;
    }
    
    /**
     * Integrates the given function on the bounds [0, infinity)
     * @return 
     */
    public static double infiniteIntegral(DoubleUnaryOperator func) 
    {
        double intVal = integrate((double t) -> infiniteToFiniteIntegralTransform(func, t), 0, 1);
        assert Double.isFinite(intVal);
        return intVal;
    }

}
