package src.GraphNetwork.Local;

import java.util.Random;
import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleBiFunction;
import java.util.function.ToDoubleFunction;

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
        this.mean = mean;
        this.variance = variance;
        N = 1;
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

    /**
     * Update the mean and variance of this distribution to accomodate the addition of a new point.
     * The new mean and variance are computed using newtons method to minimize the log-likelihood function
     * @param x the point to add to the distribution
     * @param b indicates whether the point reinforces the distribution (b=true) or diminishes the distribution (b=false)
     * @param weight the point's weight
     */
    private void newtonUpdateMeanAndVariance(double x, boolean b, double weight)
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

    private double shiftGuess(double x, double weight)
    {
        return (x - mean)/(zeta_3halfs*N*variance*root_2pi/weight + 1);
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


    private double invWeight(double x)
    {
        return 1/(1 - Math.exp(-x*x/2));
    }

    private double shiftDeltaNewton(double x, boolean b, double shift, double scale, double weight)
    {
        double d = x - mean - shift; // d = shifted distance from mean
        double scale2 = scale*scale;
        double var2 = variance*variance;

        DoublePair dp = shiftLERP(shift, scale*variance);
        if(b)
        {
            return (dp.x1 - d*weight/N)/(dp.x2 + 1d/N);
        }

        double inv_weight = invWeight(d/(variance*scale));
        double dynamic = d*(1 - inv_weight)*weight/N;
        double dynamic_deriv = (1 - inv_weight)*(d*d*inv_weight/(scale2*var2) - 1)*weight/N;
        return (dp.x1 - dynamic)/(dp.x2 - dynamic_deriv);
    }  

    private double scaleDeltaNewton(double x, boolean b, double shift, double scale, double weight)
    {
        double d = x - mean - shift; // d = distance from mean
        double d2 = d*d;
        double scale2 = scale*scale;
        double var2 = variance*variance;

        DoublePair dp = scaleLERP(scale*variance);
        if(b)
        {
            return (dp.x1 + d2*weight/(N*scale*scale2))/dp.x2;
        }

        double inv_weight = invWeight(d/(variance*scale));
        double dynamic = d2*weight*(1 - inv_weight)/(N*var2*variance);
        double dynamic_deriv = dynamic*d2*inv_weight/(scale*scale2*var2);
        return (dp.x1 + dynamic)/(dp.x2 + dynamic_deriv);
    }  

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
