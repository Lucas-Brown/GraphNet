package src.GraphNetwork.Local;

import java.util.ArrayList;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;
import java.util.stream.IntStream;

import src.NetworkTraining.LinearInterpolation2D;
import src.NetworkTraining.LinearRange;
import src.NetworkTraining.Range;

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

    private static final double w_domain = 20; // pre-compute w values on [-shift_domain, shift_domain]
    private static final double eta_domain = 10; // pre-compute eta values on (0, scale_domain]
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
     * TODO: consider non-linear maps, especially for the scale factor
     */
    static{
        Range w_range = new LinearRange(-w_domain, w_domain, w_divisions, true, true);
        Range eta_range = new LinearRange(0, eta_domain, eta_divisions, false, true);


        ToDoubleBiFunction<Double, Double> shiftFunction = (w, eta) -> infiniteIntegral((double x) -> ShiftIntegrand(x, w, eta));
        ToDoubleBiFunction<Double, Double> scaleFunction = (w, eta) -> infiniteIntegral((double x) -> ScaleIntegrand(x, w, eta));

        shiftMap = new LinearInterpolation2D(w_range, eta_range, shiftFunction);
        scaleMap = new LinearInterpolation2D(w_range, eta_range, scaleFunction);
    }

    // The distribution that is being updated
    private final BellCurveDistribution parentDistribution;

    // The mean value of the parent distribution
    private double mean;

    // The variance of the parent distribution
    private double variance;

    // The square of the variance
    private double var2;

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

        return (root_pi/var2 * varShiftSum + pointSum) / (root_pi*parentDistribution.getN() + root_pi/var2 *varSum + pointCount);
    }

    private double scaleGuess(double x, boolean reinforce, double weight)
    {
        double d = x - mean;
        double d2 = d*d;
        double var3 = variance*variance*variance;

        // this hurts
        if(reinforce)
        {
            double a = (5*zeta_7halfs - 7*zeta_5halfs)/2;
            double b = zeta_5halfs;
            double c = weight*d2/(3*root_2pi*N*var3);

            return 1 + (b - Math.sqrt(b*b - 4*a*c))/(2*a);
        }
        else
        {
            return 1 - weight*d2/(3*N*root_2pi*var3*zeta_5halfs)*invWeight(d/variance);
        }
    }


    /**
     * Update the mean and variance of this distribution to accomodate the addition of a new point.
     * The new mean and variance are computed using newtons method to minimize the log-likelihood function
     * @param x the point to add to the distribution
     * @param b indicates whether the point reinforces the distribution (b=true) or diminishes the distribution (b=false)
     * @param weight the point's weight
     */
    private void newtonUpdate(double x, boolean b, double weight)
    {
        double shift = shiftGuess(x, weight);
        double scale = scaleGuess(x, b, weight);

        double delta_shift;
        double delta_scale;
        do{
            delta_shift = shiftDeltaNewton(x, b, shift, scale, weight);
            delta_scale = scaleDeltaNewton(x, b, shift, scale, weight);
            shift -= delta_shift;
            scale -= delta_scale;
        }while(Math.abs(delta_shift) > TOLLERANCE || Math.abs(delta_scale) > TOLLERANCE);

        mean += shift;
        variance *= scale;
        N += weight;
    }

// End newton's method
    
    /**
     * compute the relative shift parameter "w"
     * @param bcd
     * @return
     */
    private double getRelativeShift(BellCurveDistribution bcd)
    {
        return (mean - bcd.getMeanValue() - shift) / (root_2 * scale * variance);
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
        double net = parentDistribution.getN() * shiftResidueDistribution(parentDistribution);

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
        double scaled_var2 = variance * shift;
        scaled_var2 *= scaled_var2;
        if(!b)
        {
            net *= 1/Math.expm1(-d*d/(2*scaled_var2)); 
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
        double net = parentDistribution.getN() * shiftDerivativeResidueDistribution(parentDistribution);

        // contribution from individual points
        net += IntStream.range(0, update_points.size())
            .mapToDouble(i -> shiftDerivativeResiduePoint(update_points.get(i), points_b.get(i), point_weights.get(i)))
            .sum();


        // contribution from distributions
        net += influincingDistributions.stream().mapToDouble(this::shiftResidueDistribution).sum();

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
        double scaled_var2 = variance * shift;
        scaled_var2 *= scaled_var2;

        final double inv_expm1 = 1/Math.expm1(-d*d/(2*scaled_var2));
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
        return getShiftParameterDerivative(w, eta, bcd.getVariance());
    }

    
    /**
     * Returns the derivative of the interpolated shift parameter 
     * @param w
     * @param eta
     * @param sigma_b
     * @param n
     * @return
     */
    private double getShiftParameterDerivative(double w, double eta, double sigma_b)
    {
        final double coef = root_2*sigma_b*eta/root_pi;
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
        double net = parentDistribution.getN() * scaleResidueDistribution(parentDistribution, 1);

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
        double scaled_var2 = variance * shift;
        scaled_var2 *= scaled_var2;
        if(!b)
        {
            net *= 1/Math.expm1(-d2/(2*scaled_var2)); 
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
        final double coef = sigma_b*sigma_b*Math.pow(eta, 3d/2);
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
        double net = parentDistribution.getN() * shiftDerivativeResidueDistribution(parentDistribution);

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

        final double inv_expm1 = 1/Math.expm1(-d*d/(2*scaled_var2));
        return d*(1 - inv_expm1) * (d*d*inv_expm1/scaled_var2 - 2);
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
        // incrementally build the derivative
        double deriv = -zeta_3halfs/B;
        deriv += 2/root_pi * scaleMap.interpolate(w, eta);
        deriv += 4/(3*root_pi) * eta * scaleMap.interpolateDerivative(w, eta)[1]; // index 1 = derivative with respect to the second parameter (eta)
        return deriv * 3 * variance * sigma_b * eta;
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
        // Limit as x -> 0 
        if(x == 0)
        {
            return -4 * eta * w * Math.exp(-eta*w*w);
        }

        // As x becomes large, the shift integrand (y) approaches 0
        // once y = 0.001, approximate y = 0 to reduce computation and avoid inf/nan in later steps
        if(x > 1.17083311923 * w + 2.18931364315)
        {
            return 0;
        }

        final double wminx = w-x;
        final double wplusx = w+x;
        double value = Math.exp(-eta * wminx * wminx) - Math.exp(-eta * wplusx * wplusx);
        value *= x;

        // beyond x > 3, the value of Math.expm1(-x*x) is approximately 1
        if(x > 3)
        {
            return value; 
        }
        else
        {
            return value / Math.expm1(-x*x);
        }
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
        if(x == 0)
        {
            return 2 * Math.exp(-eta*w*w);
        }

        // As x becomes large, the scale integrand (y) approaches 0
        // once y = 0.001, approximate y = 0 to reduce computation and avoid inf/nan in later steps
        if(x > 1.41205967868 * w + 2.43434774198)
        {
            return 0;
        }

        final double wminx = w-x;
        final double wplusx = w+x;
        double value = Math.exp(-eta * wminx * wminx) + Math.exp(-eta * wplusx * wplusx);
        value *= x*x;

        // beyond x > 3, the value of Math.expm1(-x*x) is approximately 1
        if(x > 3)
        {
            return value; 
        }
        else
        {
            return - value / Math.expm1(-x*x);
        }
    }

    /**
     * Use the transformation x = e^(t/(1-t)) - 1 to convert an integral from the bounds [0, Infinity) to [0, 1]
     * @param func the function being integrated over. MUST converge EXPONENTIALLY to 0 as x -> infinity
     * @param t the evaluation point on the bounds [0, 1]
     * @return the transformed value at the given point
     */
    private static double infiniteToFiniteIntegralTransform(DoubleUnaryOperator func, double t)
    {
        final double temp = 1d/(1-t);
        final double x = Math.expm1(t*temp); 
        // if x is effectively infinite, then the provided function is assumed to have a value of 0 due to convergence requirement
        if(Double.isInfinite(x))
        {
            return 0;
        }
        final double transformedIntegral = func.applyAsDouble(x) * (x + 1) * temp*temp;
        
        assert Double.isFinite(transformedIntegral);
        return transformedIntegral;
    }

    /**
     * Integrate a function on the bounds of 0 to 1
     * @param func
     * @return
     */
    private static double integrate(DoubleUnaryOperator func)
    {
        Range t_range = new LinearRange(0, 1, integrationDivisions-2, false, false);
        double intermediate = t_range.stream().map(func).sum()/integrationDivisions;
        return intermediate + (func.applyAsDouble(0) + func.applyAsDouble(1)) / 2;
    }
    
    /**
     * Integrates the given function on the bounds [0, infinity)
     * @return 
     */
    private static double infiniteIntegral(DoubleUnaryOperator func) 
    {
        double intVal = integrate((double t) -> infiniteToFiniteIntegralTransform(func, t));
        assert Double.isFinite(intVal);
        return intVal;
    }





}
