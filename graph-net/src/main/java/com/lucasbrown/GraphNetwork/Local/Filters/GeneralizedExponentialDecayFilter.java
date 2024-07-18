package com.lucasbrown.GraphNetwork.Local.Filters;

import java.util.Random;

import com.lucasbrown.HelperClasses.MathHelpers;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;

import static com.lucasbrown.HelperClasses.MathHelpers.sigmoid;
import static com.lucasbrown.HelperClasses.MathHelpers.sigmoid_derivative;

/**
 * f\left(x,a,b,u,s,n\right)=\left(b-a\right)e^{-\frac{1}{2}\left|\frac{x-u}{s}\right|^{n}}+a
 */
public class GeneralizedExponentialDecayFilter implements IFilter{

    private final Random rng;

    // parameters
    private double lower_param, upper_param, mean, variance, power; 

    public GeneralizedExponentialDecayFilter(double lower_param, double upper_param, double mean, double variance, double power, Random random){
        this.lower_param = lower_param;
        this.upper_param = upper_param;
        this.mean = mean;
        this.variance = variance;
        this.power = power;
        rng = random;
    }

    public GeneralizedExponentialDecayFilter(double lower_param, double upper_param, double mean, double variance, double power)
    {
        this(lower_param, upper_param, mean, variance, power, new Random());
    }

    @Override
    public boolean shouldSend(double x) {
        return rng.nextDouble() > getChanceToSend(x);
    }

    @Override
    public double getChanceToSend(double x) {
        double w = Math.abs((x - mean)/variance);
        double lower = sigmoid(lower_param);
        double upper = sigmoid(upper_param);
        return (upper-lower)*Math.exp(-Math.pow(w, power)/2) + lower;
    }

    @Override
    public void applyAdjustments(IExpectationAdjuster adjuster) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'applyAdjustments'");
    }

    @Override
    public int getNumberOfAdjustableParameters() {
        return 5;
    }

    @Override
    public double[] getAdjustableParameters() {
        return new double[]{lower_param, upper_param, mean, variance, power};
    }

    @Override
    public void setAdjustableParameters(double... params) {
        assert params.length == 5;
        lower_param = params[0];
        upper_param = params[1];
        mean = params[2];
        variance = params[3];
        power = params[4];
    }

    @Override
    public void applyAdjustableParameterUpdate(double[] delta) {
        lower_param -= delta[0];
        upper_param -= delta[1];
        mean -= delta[2];
        variance -= delta[3];
        power -= delta[4];
    }

    private double[] getDerivativeOfParameters(double x){
        double w = Math.abs((x - mean)/variance);
        double lower = sigmoid(lower_param);
        double upper = sigmoid(upper_param);
        double w_pow = Math.pow(w, power);
        double exp = Math.exp(-w_pow/2);
        double range = upper - lower;
        double d_exp = range*exp*w_pow/2;

        double d_lower = (1-exp)*sigmoid_derivative(lower_param);
        double d_upper = exp*sigmoid_derivative(upper_param);
        double d_mean = (x - mean) == 0 ? 0 : d_exp*power/(x-mean);
        double d_variance = d_exp*power / variance;
        double d_power = w == 0 ? 0 : -d_exp*Math.log(w);
        return new double[]{d_lower, d_upper, d_mean, d_variance, d_power};
    }

    @Override
    public double[] getLogarithmicParameterDerivative(double x) {
        double eval = getChanceToSend(x);
        double[] derivatives = getDerivativeOfParameters(x);
        for (int i = 0; i < derivatives.length; i++) {
            derivatives[i] /= eval;
        }
        return derivatives;
    }

    @Override
    public double[] getNegatedLogarithmicParameterDerivative(double x) {
        double eval = getChanceToSend(x);
        double[] derivatives = getDerivativeOfParameters(x);
        for (int i = 0; i < derivatives.length; i++) {
            derivatives[i] /= eval-1;
        }
        return derivatives;
    }

    @Override
    public double getLogarithmicDerivative(double x) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getLogarithmicDerivative'");
    }

    @Override
    public double getNegatedLogarithmicDerivative(double x) {
        // TODO Auto-generated method stub
        throw new UnsupportedOperationException("Unimplemented method 'getNegatedLogarithmicDerivative'");
    }

    public static GeneralizedExponentialDecayFilter getEvenChanceDistribution(){
        return new GeneralizedExponentialDecayFilter(0, 0, 0, 1, 2);
    }
    
}
