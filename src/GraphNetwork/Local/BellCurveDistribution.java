package src.GraphNetwork.Local;

import java.util.Random;

import src.GraphNetwork.Global.DoublePair;

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

    private static final double shift_domain = 10; // pre-compute shift values on [-shift_domain, shift_domain]
    private static final double scale_domain = 10; // pre-compute scale values on (0, scale_domain]
    private static final int shift_divisions = 100;
    private static final int scale_divisions = 100; 

    private static final double[][] shiftMap;
    private static final double[][] shiftDerivativeMap;
    private static final double[] scaleMap;
    private static final double[] scaleDerivativeMap;

    /**
     * Construct the linear interpolation maps.
     * TODO: consider non-linear maps, especially for the scale factor
     */
    static{
        shiftMap = new double[shift_divisions][scale_divisions];
        shiftDerivativeMap = new double[shift_divisions][scale_divisions];
        scaleMap = new double[scale_divisions];
        scaleDerivativeMap = new double[scale_divisions];

        // loop over all scale values
        for(int i = 0; i < shift_divisions; i++)
        {
            double scale = scale_domain*(1d-((double) i)/scale_divisions);
            
            scaleMap[i] = scaleResidual(scale);
            scaleDerivativeMap[i] = scaleResidualDerivative(scale);

            // loop over all shift values
            for(int j = 0; j < scale_divisions; j++)
            {
                double shift = shift_domain*(2d * j / (shift_divisions - 1d) - 1d);

                shiftMap[i][j] = shiftResidual(shift, scale);
                shiftDerivativeMap[i][j] = shiftResidualDerivative(shift, scale);

            }
        }

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
    public boolean shouldSend(double inputSignal, double factor) {
        // Use the normalized normal distribution as a measure of how likely  
        return factor*computeNormalizedDist(inputSignal) >= rand.nextDouble();
    }

    @Override
    public double getDirectionOfDecreasingLikelyhood(double x)
    {
        return x > mean ? 1 : -1;
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
     * Compute the residual of the shift factor
     * @param shift 
     * @param scale
     * @return
     */
    private static double shiftResidual(double shift, double scale)
    {
        double exponent_value = -shift*shift/(2*root_pi*scale*scale);
        return shift*scale*root_2pi*Math.pow(zeta_3halfs, Math.exp(exponent_value)); // yup... a double exponential for some reason. This was approximated without proof
    }

    /**
     * Compute the derivative (wrt the shift) of the residual of the shift factor 
     * @param shift 
     * @param scale
     * @return
     */
    private static double shiftResidualDerivative(double shift, double scale)
    {
        if(shift == 0) return scale*root_2pi*zeta_3halfs;

        double res = shiftResidual(shift, scale);
        // build up the derivative in steps
        double derivative = Math.log(res/(shift*scale*root_2pi));
        derivative *= -shift/(scale*scale*root_pi);
        derivative += 1/shift;
        return derivative * res;
    }

    /**
     * Compute the residual of the scale factor
     * @param scale
     * @return
     */
    private static double scaleResidual(double scale)
    {
        return root_2pi*scale*scale*scale*(hurwitz_zeta(3d/2, scale*scale) - zeta_3halfs);
    }

    /**
     * Compute the derivative (wrt the scale) of the residual of the scale factor 
     * @param scale
     * @return
     */
    private static double scaleResidualDerivative(double scale)
    {
        double scale2 = scale*scale;
        return 3*root_2pi*scale2*(hurwitz_zeta(3d/2, scale2) - scale2*hurwitz_zeta(5d/2, scale2) - zeta_3halfs);
    }

    private static DoublePair shiftLERP(double shift, double scale)
    {
        double i_float = scale_divisions*(1d-scale/scale_domain);
        double j_float = (shift/shift_domain + 1d)*(shift_divisions - 1d)/2d;
        
        // if the shift or scale is outside of the pre-computed range, recompute the value
        if(i_float < 0 || i_float > scale_divisions || j_float < 0 || j_float > shift_divisions)
        {
            return new DoublePair(shiftResidual(shift, scale), shiftResidualDerivative(shift, scale));
        }

        int i = (int)i_float;
        int j = (int)j_float;
        double wi = i == i_float ? 1 : i + 1 - i_float;
        double wj = j == j_float ? j : j + 1 - j_float;

        double w11 = wi*wj;
        double w12 = wi*(1-wj);
        double w21 = (1-wi)*wj;
        double w22 = (1-wi)*(1-wj);

        double res = w11*shiftMap[i][j]+w12*shiftMap[i][j+1]+w21*shiftMap[i+1][j]+w22*shiftMap[i+1][j+1];
        double resD = w11*shiftDerivativeMap[i][j]+w12*shiftDerivativeMap[i][j+1]
                     +w21*shiftDerivativeMap[i+1][j]+w22*shiftDerivativeMap[i+1][j+1];
        return new DoublePair(res, resD);
    }

    private static DoublePair scaleLERP(double scale)
    {
        double i_float = scale_divisions*(1d-scale/scale_domain);
        
        // if the scale is outside of the pre-computed range, recompute the value
        if(i_float < 0 || i_float > scale_divisions)
        {
            return new DoublePair(scaleResidual(scale), scaleResidualDerivative(scale));
        }

        int i = (int)i_float;
        double w = i + 1 - i_float;
        if(w == 1)
        {
            return new DoublePair(scaleMap[i], scaleDerivativeMap[i]);
        }
        else
        {
            double res = w*scaleMap[i]+(1-w)*scaleMap[i+1];
            double resD = w*scaleDerivativeMap[i]+(1-w)*scaleDerivativeMap[i+1];
            return new DoublePair(res, resD);
        }
    }


}
