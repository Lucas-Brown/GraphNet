package com.lucasbrown.GraphNetwork.Local;

import java.util.ArrayList;
import java.util.function.DoubleUnaryOperator;
import java.util.stream.IntStream;

import com.lucasbrown.NetworkTraining.LinearInterpolation2D;
import com.lucasbrown.NetworkTraining.LinearRange;
import com.lucasbrown.NetworkTraining.MultiplicitiveRange;
import com.lucasbrown.NetworkTraining.Range;

import jsat.math.Function;


public class BellCurveDistributionAdjuster {
    
    private static final double TOLLERANCE = 1E-6; // tollerance for newton's method  

    /**
     * pre-compute a few values of the riemann-zeta function
     */
    public static final double zeta_3halfs = riemann_zeta(3d/2);
    public static final double zeta_5halfs = riemann_zeta(5d/2);
    public static final double zeta_7halfs = riemann_zeta(7d/2);
    public static final double root_pi = Math.sqrt(Math.PI);
    public static final double root_2 = Math.sqrt(2d);
    public static final double root_2pi = root_pi*root_2;

    private static final double w_domain = 10; // pre-compute w values on [-shift_domain, shift_domain]
    private static final double eta_domain = 10; // pre-compute eta values on [1/scale_domain, scale_domain]
    private static final int w_divisions = 500;
    private static final int eta_divisions = 500; 
    private static final int integrationDivisions = 1000;

    /**
     * shift map dimensions are computed as [w][eta]
     * where eta = lambda^2 * (sigma_0)^2 / (sigma_b)^2 
     * and w = (u_0 - u_b + a) / (lambda * sigma_0 * sqrt(2)) 
     * 
     * using a as the shift parameter and lambda as the scale parameter
     */
    private static final LinearInterpolation2D shiftMap;
    private static final LinearInterpolation2D scaleMap;

    /**
     * Construct the linear interpolation maps.
     */
    static{
        Range w_range = new LinearRange(-w_domain, w_domain, w_divisions, true, true);
        Range eta_range = new MultiplicitiveRange(1/eta_domain, eta_domain, eta_divisions, true, true);

        shiftMap = new LinearInterpolation2D(w_range, eta_range, BellCurveDistributionAdjuster::shiftFunctionNoTransform);
        scaleMap = new LinearInterpolation2D(w_range, eta_range, BellCurveDistributionAdjuster::scaleFunctionNoTransform);
    }

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

    /**
        The filter amplitudes of all corresponding influicing distributions.  
        This is not a traditional weight, but rather, is intended to indicate the influence from convolutions of distributions as filters
    */
    private ArrayList<Double> distribution_weights;

    // The x-values of all points being used to reinforce/diminish the distribution 
    private ArrayList<Double> update_points;

    // The reinforcement value for each point. I.E b = true for reinforcement, b = false for diminishment
    private ArrayList<Boolean> points_b;

    // The weight of each point
    private ArrayList<Double> point_weights;

    // The shift value. Used during iterative updating
    private double shift;

    // The scale value. Used during iterative updating
    private double scale;


public BellCurveDistributionAdjuster(BellCurveDistribution parentDistribution)
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
// Begin Newton's method 
    
/**
 * Estimates the new shift value
 * @return
 */
    private double shiftGuess()
    {
        double mu_0 = parentDistribution.getMeanValue();
        double var2 = parentDistribution.getVariance() * parentDistribution.getVariance();

        double pointSum = IntStream.range(0, update_points.size())
            .filter(points_b::get)
            .mapToDouble(i -> update_points.get(i) - mean)
            .sum();

        long pointCount = points_b.stream().filter(b -> b).count();

        double varSum = 0;
        double varShiftSum = 0;
        for(BellCurveDistribution bcd : influincingDistributions)
        {
            double sigma_b2 = bcd.getVariance() * bcd.getVariance();
            varSum += sigma_b2;
            varShiftSum += sigma_b2 * (bcd.getMeanValue() - mu_0);
        }

        return (root_pi/var2 * varShiftSum + pointSum) / (root_pi*N + root_pi/var2 *varSum + pointCount);
    }

    private double scaleGuess()
    {
        // Use a cubic approximation for distributions and a quadratic approximation for points
        // Start by iteratively constructing the coefficents

        // ax^3 + bx^2 + c = 0
        double a = 0;
        double b = 0;
        double c = 0;

        // parent distribution contribution
        double deriv = getScaleParameterDerivative(0, 1, variance, 1)/3;
        c += N *(getScaleParameter(0, 1, variance, 1) - deriv);
        a += N * deriv;

        // individual points
        for(int i = 0; i < update_points.size(); i++)
        {
            deriv = scaleDerivativeResiduePoint(update_points.get(i), points_b.get(i), point_weights.get(i))/2;
            b += deriv;
            c += scaleResiduePoint(update_points.get(i), points_b.get(i), point_weights.get(i)) - deriv;
        }

        // distributions
        for(int i = 0; i < influincingDistributions.size(); i++)
        {
            deriv = scaleDerivativeResidueDistribution(influincingDistributions.get(i), distribution_weights.get(i))/3;
            b += deriv;
            c += scaleResidueDistribution(influincingDistributions.get(i), distribution_weights.get(i)) - deriv;
        }

        return solveSimpleCubic(a, b, c);
    }


    /**
     * Update the mean and variance using newtons method
     */
    public void newtonUpdate()
    {
        // set initial values to make "getRelativeShift" and "getRelativeScale" return proper values
        shift = 0;
        scale = 1;

        // get a good initial guess 
        shift = shiftGuess();
        scale = scaleGuess(); 

        double delta_shift;
        double delta_scale;
        do{
            delta_shift = netShiftResidue()/netShiftDerivativeResidue();
            delta_scale = netScaleResidue()/netScaleDerivativeResidue();
            shift -= delta_shift;
            scale -= delta_scale;
        }while(Math.abs(delta_shift) > TOLLERANCE || Math.abs(delta_scale) > TOLLERANCE);

        mean += shift;
        variance *= scale;

        N += update_points.size() + influincingDistributions.size();
        clear();
    }

    
    /**
     * Update the mean and variance using nelder-mead method
     */
    public void nelderMeadUpdate()
    {
        // set initial values to make "getRelativeShift" and "getRelativeScale" return proper values
        shift = 0;
        scale = 1;

        Function toMaximize = new DoubleFunction()

        // get a good initial guess 
        shift = shiftGuess();
        scale = scaleGuess(); 

        double delta_shift;
        double delta_scale;
        do{
            delta_shift = netShiftResidue()/netShiftDerivativeResidue();
            delta_scale = netScaleResidue()/netScaleDerivativeResidue();
            shift -= delta_shift;
            scale -= delta_scale;
        }while(Math.abs(delta_shift) > TOLLERANCE || Math.abs(delta_scale) > TOLLERANCE);

        mean += shift;
        variance *= scale;

        N += update_points.size() + influincingDistributions.size();
        clear();
    }

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

// End newton's method
    
    /**
     * compute the relative shift parameter "w"
     * @param bcd
     * @return
     */
    private double getRelativeShift(BellCurveDistribution bcd)
    {
        return (mean - bcd.getMeanValue() + shift) / (root_2 * scale * variance);
    }

    /**
     * Compute the relative scale parameter "eta"
     * @param bcd
     * @return
     */
    private double getRelativeScale(BellCurveDistribution bcd)
    {
        double eta = scale * variance / bcd.getVariance();
        return eta * eta;
    }

// Begin shift

    /**
     * Compute the total residue for the shift parameter
     * @return
     */
    private double netShiftResidue()
    {
        // contribution from parent distribution
        double net = N * shiftResidueDistribution(parentDistribution);

        // contribution from individual points
        net += IntStream.range(0, update_points.size())
            .mapToDouble(i -> shiftResiduePoint(update_points.get(i), points_b.get(i), point_weights.get(i)))
            .sum();

        // contribution from distributions
        net += influincingDistributions.stream().mapToDouble(this::shiftResidueDistribution).sum();

        return net;
    }

    /**
     * Compute the net shift residue for a fixed point and a set of weighted gaussians
     * @param x the value of the fixed point
     * @param b whether the fixed point is reinforcing 
     * @param weight the weight of the fixed point to the distribution
     * @return
     */
    private double shiftResiduePoint(double x, boolean b, double weight)
    {
        double d = x - mean - shift;
        double net = weight * d;
        
        if(d == 0)
        {
            return net;
        }

        double scaled_var2 = variance * shift;
        scaled_var2 *= scaled_var2;
        if(!b)
        {
            net *= 1 + 1/Math.expm1(-d*d/(2*scaled_var2)); 
        }

        return net;
    }  

    /**
     * Returns the interpolated shift residue for a distribution
     * @param bcd
     * @return
     */
    private double shiftResidueDistribution(BellCurveDistribution bcd)
    {
        final double w = getRelativeShift(bcd);
        final double eta = getRelativeScale(bcd);
        return getShiftParameter(w, eta, bcd.getVariance());
    }

    /**
     * Returns the interpolated shift parameter 
     * @param w
     * @param eta
     * @param sigma_b
     * @return
     */
    private double getShiftParameter(double w, double eta, double sigma_b)
    {
        final double coef = root_2*sigma_b*eta/root_pi;
        return coef * shiftMap.interpolate(w, eta);
    }

// End shift
// Start shift derivative

    /**
     * Compute the total residue for the derivative of the shift parameter
     * @return
     */
    private double netShiftDerivativeResidue()
    {
        // contribution from parent distribution
        double net = N * shiftDerivativeResidueDistribution(parentDistribution);

        // contribution from individual points
        net += IntStream.range(0, update_points.size())
            .mapToDouble(i -> shiftDerivativeResiduePoint(update_points.get(i), points_b.get(i), point_weights.get(i)))
            .sum();


        // contribution from distributions
        net += influincingDistributions.stream().mapToDouble(this::shiftDerivativeResidueDistribution).sum();

        return net;
    }

    /**
     * Compute the shift residue derivative for a fixed point
     * @param x the value of the fixed point
     * @param b whether the fixed point is reinforcing 
     * @param weight the weight of the fixed point to the distribution
     * @return
     */
    private double shiftDerivativeResiduePoint(double x, boolean b, double weight)
    {
        if(b)
        {
            return -weight;
        }

        final double d = x - mean - shift;
        if(d == 0)
        {
            return 0;
        }

        double scaled_var2 = variance * shift;
        scaled_var2 *= scaled_var2;

        final double inv_expm1 = -1/Math.expm1(-d*d/(2*scaled_var2));
        return weight * (1 - inv_expm1) * (d*d*inv_expm1/scaled_var2 - 1);
    }  

    /**
     * Compute the shift residue derivative for a distribution
     * @return
     */
    private double shiftDerivativeResidueDistribution(final BellCurveDistribution bcd)
    {
        final double w = getRelativeShift(bcd);
        final double eta = getRelativeScale(bcd);
        return getShiftParameterDerivative(w, eta);
    }

    
    /**
     * Returns the derivative of the interpolated shift parameter 
     * @param w
     * @param eta
     * @param n
     * @return
     */
    private double getShiftParameterDerivative(double w, double eta)
    {
        final double coef = Math.sqrt(eta)/root_pi;
        return coef * shiftMap.interpolateDerivative(w, eta)[0];
    }

// End shift derivative
// Begin scale


    /**
     * Compute the total reside for the scale parameter
     */
    private double netScaleResidue()
    {
        // contribution from parent distribution
        double net = N * scaleResidueDistribution(parentDistribution, 1);

        // contribution from individual points
        net += IntStream.range(0, update_points.size())
            .mapToDouble(i -> scaleResiduePoint(update_points.get(i), points_b.get(i), point_weights.get(i)))
            .sum();


        // contribution from distributions
        net += IntStream.range(0, influincingDistributions.size())
            .mapToDouble(i -> scaleResidueDistribution(influincingDistributions.get(i), distribution_weights.get(i)))
            .sum();
            
        return net;
    }  

    /**
     * Compute the scale residue for a fixed point
     * @param x the value of the fixed point
     * @param b whether the fixed point is reinforcing 
     * @param weight the weight of the fixed point to the distribution
     * @return
     */
    private double scaleResiduePoint(double x, boolean b, double weight)
    {
        double d2 = x - mean - shift;
        d2 *= d2;
        double net = weight * d2;

        if(d2 == 0)
        {
            return net;
        }

        double scaled_var2 = variance * shift;
        scaled_var2 *= scaled_var2;
        if(!b)
        {
            net *= (1 + 1/Math.expm1(-d2/(2*scaled_var2))); 
        }

        return net;
    }  

    /**
     * Compute the scale residue for a distribution
     * @param bcd
     * @param B
     * @return
     */
    private double scaleResidueDistribution(final BellCurveDistribution bcd, final double B)
    {
        final double w = getRelativeShift(bcd);
        final double eta = getRelativeScale(bcd);
        return getScaleParameter(w, eta, bcd.getVariance(), B);
    }
    
    /**
     * Returns the interpolated scale parameter 
     * @param w
     * @param eta
     * @param sigma_b
     * @param B The amplitude coeficient to the weight approximation 
     * @return
     */
    private double getScaleParameter(double w, double eta, double sigma_b, double B)
    {
        final double root_eta = Math.sqrt(eta);
        final double coef = sigma_b*sigma_b*root_eta*root_eta*root_eta; // eta^(3/2)
        return coef * (2/root_pi * scaleMap.interpolate(w, eta) - zeta_3halfs/B);
    }

// End scale
// Begin scale derivative

    /**
     * Compute the total reside for the scale parameter
     */
    private double netScaleDerivativeResidue()
    {
        // contribution from parent distribution
        double net = N * scaleDerivativeResidueDistribution(parentDistribution, 1);

        // contribution from individual points
        net += IntStream.range(0, update_points.size())
            .mapToDouble(i -> scaleDerivativeResiduePoint(update_points.get(i), points_b.get(i), point_weights.get(i)))
            .sum();

        // contribution from distributions
        net += IntStream.range(0, influincingDistributions.size())
            .mapToDouble(i -> scaleDerivativeResidueDistribution(influincingDistributions.get(i), distribution_weights.get(i)))
            .sum();
            
        return net;
    }  

    /**
     * Compute the scale residue derivative for a fixed point 
     * @param x the value of the fixed point
     * @param b whether the fixed point is reinforcing 
     * @param weight the weight of the fixed point to the distribution
     * @return
     */
    private double scaleDerivativeResiduePoint(double x, boolean b, double weight)
    {
        if(b)
        {
            return -2*weight*x;
        }

        final double d = x - mean - shift;
        double scaled_var2 = variance * shift;
        scaled_var2 *= scaled_var2;

        final double inv_expm1 = -1/Math.expm1(-d*d/(2*scaled_var2));
        return weight*d*(1 - inv_expm1) * (d*d*inv_expm1/scaled_var2 - 2);
    }  

    /**
     * See other netScaleResidueDerivative
     * @param mu_arr
     * @param sigma_arr
     * @return
     */
    private double scaleDerivativeResidueDistribution(final BellCurveDistribution bcd, final double B)
    {
        final double w = getRelativeShift(bcd);
        final double eta = getRelativeScale(bcd);
        return getScaleParameterDerivative(w, eta, bcd.getVariance(), B);
    }

    
    /**
     * Returns the derivative of the interpolated scale parameter 
     * @param w
     * @param eta
     * @param sigma_b
     * @param B The amplitude coeficient to the weight approximation 
     * @param n
     * @return
     */
    private double getScaleParameterDerivative(double w, double eta, double sigma_b, double B)
    {
        double[] w_eta_derivatives = scaleMap.interpolateDerivative(w, eta);

        // incrementally build the derivative with respect to the scale parameter
        // non-derivative part
        double deriv = 3*(scaleMap.interpolate(w, eta) - root_pi*zeta_3halfs/(2*B));
        // derivative part for w
        deriv -= w * w_eta_derivatives[0];

        // derivative part for eta
        deriv += eta * w_eta_derivatives[1];

        // all multiplied by a coefficient
        return 2*variance*sigma_b*eta*deriv/root_pi;
    }

// End scale derivative
// Begin static pre-computation helpers

    public static double hurwitz_zeta(double s, double a)
    {
        double sum = 0;
        int k = 0;
        double delta;
        do{
            delta = Math.pow(a + k++, -s);
            sum += delta;
        }while(delta/sum > 1E-8);
        return sum;
    }

    public static double riemann_zeta(double s)
    {
        return hurwitz_zeta(s, 1);
    }

    /**
     * Compute the value of the shift integrand at a given point
     * @param x The x-value to evaluate. x indicates the variable to be integrated over from 0 to infinity
     * @param w The shift parameter
     * @param eta The scale parameter 
     * @return The integrand value
     */
    private static double ShiftIntegrand(double x, double w, double eta)
    {
        // special case
        if(w == 0)
        {
            return 0;
        }

        // Use series expansion around x=0 to avoid infinities from expm1 
        if(x <= 0.1)
        {
            double temp = -4 - 2/3*(4*eta*eta*w*w - 6*eta + 3) * x*x;
            return temp * w * eta * Math.exp(-eta*w*w);
        }

        double w_abs = Math.abs(w);

        // As x becomes large, the shift integrand approaches 0
        if(x >= w_abs + Math.sqrt(Math.log(w_abs/TOLLERANCE)/eta))
        {
            return 0;
        }
        
        final double wminx = w-x;
        final double wplusx = w+x;


        double value = Math.exp(-eta * wplusx * wplusx) - Math.exp(-eta * wminx * wminx);
        value *= x;
        return - value / Math.expm1(-x*x);

        /* 
        if(x >= Math.sqrt(Math.log(1/TOLLERANCE)))
        {
            return value; 
        }
        else
        {
            return - value / Math.expm1(-x*x);
        }
        */
    }

    /**
     * Compute the value of the scale integrand at a given point
     * @param x The x-value to evaluate. x indicates the variable to be integrated over from 0 to infinity
     * @param w The shift parameter
     * @param eta The scale parameter 
     * @return The integrand value
     */
    private static double ScaleIntegrand(double x, double w, double eta)
    {
        // Use series expansion around x=0 to avoid infinities from expm1 
        if(x <= 0.1)
        {
            double temp = 2 +(4*eta*eta*w*w - 2*eta + 1) * x*x;
            return temp * Math.exp(-eta*w*w);
        }

        
        double w_abs = Math.abs(w);

        // As x becomes large, the scale integrand approaches 0
        if(w_abs == 0)
        {
            if(x >= Math.sqrt(Math.log(2/TOLLERANCE)/eta))
            {
                return 0;
            }
        }
        else
        {
            if(x >= w_abs + Math.sqrt(Math.log(w_abs*w_abs/TOLLERANCE)/eta))
            {
                return 0;
            }
        }

        final double wminx = w-x;
        final double wplusx = w+x;
        double value = Math.exp(-eta * wplusx * wplusx) + Math.exp(-eta * wminx * wminx);
        value *= x*x;
        value /= -Math.expm1(-x*x);
        if(!Double.isFinite(value)) 
        {
            System.err.println();
        }
        return value;

        /*
        if(x > 3)
        {
            return value; 
        }
        else
        {
            return - value / Math.expm1(-x*x);
        }
         */
    }

    /**
     * Use the transformation x = e^(t/(1-t)) - 1 to convert an integral from the bounds [0, Infinity) to [0, 1]
     * @param func the function being integrated over. MUST converge EXPONENTIALLY to 0 as x -> infinity
     * @param t the evaluation point on the bounds [0, 1]
     * @return the transformed value at the given point
     */
    public static double infiniteToFiniteIntegralTransform(DoubleUnaryOperator func, double t)
    {
        final double temp = 1d/(1-t);
        final double x = Math.expm1(t*temp); 
        // if x is effectively infinite, then the provided function is assumed to have a value of 0 due to convergence requirement
        if(Double.isInfinite(x))
        {
            return 0;
        }
        final double transformedIntegral = func.applyAsDouble(x) * (x + 1) * temp*temp;
        
        if(!Double.isFinite(transformedIntegral))
        {
            System.err.println(); // just need a place to stop
        }
        assert Double.isFinite(transformedIntegral);
        return transformedIntegral;
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

    /**
     * Solve the equation a*x^3 + b*x^2 + c = 0 assuming the equation only has one real root
     * @param a
     * @param b
     * @param c
     * @return the singular x-solution
     */
    public static double solveSimpleCubic(double a, double b, double c)
    {
        // normalize constants to avoid rounding errors
        b /= a;
        c /= a;

        double b3 = b*b*b;

        // this assertion technically allows for cubics with 2 real roots: one root of order one and another of order 2
        // In this case of two roots, this function returns the root of order one.  
        assert c >= - 4/27 * b3: "Cubic has multiple real solutions";


        double radical = 3*Math.sqrt(3*(27*c*c + 4*b3*c));
        radical -= 27*c + 2*b3;
        radical = Math.cbrt(radical/2); 

        return (radical + b*b/radical - b)/3;
    }

    public static double shiftFunctionNoTransform(double w, double eta)
    {
        if(w == 0) return 0;
        double upper_bound = Math.abs(w) + Math.sqrt(Math.log(Math.abs(w)/TOLLERANCE)/eta);
        return integrate((double x) -> ShiftIntegrand(x, w, eta), 0, upper_bound);
    } 

    public static double scaleFunctionNoTransform(double w, double eta)
    {
        double upper_bound;
        if(w == 0)
        {
            upper_bound = Math.sqrt(Math.log(2/TOLLERANCE)/eta);
        }
        else
        {
            upper_bound = Math.abs(w) + Math.sqrt(Math.log(w*w/TOLLERANCE)/eta);
        }

        return integrate((double x) -> ScaleIntegrand(x, w, eta), 0, upper_bound);
    } 

    public static double shiftFunction(double w, double eta)
    {
        return infiniteIntegral((double x) -> ShiftIntegrand(x, w, eta));
    } 

    public static double scaleFunction(double w, double eta)
    {
        return infiniteIntegral((double x) -> ScaleIntegrand(x, w, eta));
    } 

    public static double getShiftIntegralValue(double w, double eta)
    {
        return shiftMap.interpolate(w, eta);
    }
    
    public static double getShiftIntegralDerivativeValue(double w, double eta)
    {
        return shiftMap.interpolateDerivative(w, eta)[0];
    }

    public static double getScaleIntegralValue(double w, double eta)
    {
        return scaleMap.interpolate(w, eta);
    }
    
    public static double getScaleIntegralDerivativeValue(double w, double eta)
    {
        return scaleMap.interpolateDerivative(w, eta)[1];
    }

}
