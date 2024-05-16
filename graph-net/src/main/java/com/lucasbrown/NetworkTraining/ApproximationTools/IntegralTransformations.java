package com.lucasbrown.NetworkTraining.ApproximationTools;

import java.util.function.DoubleUnaryOperator;

/**
 * A static helper class of various integral transformations
 */
public class IntegralTransformations {

    /**
     * Truncates an integral with bounds (-inf, inf) to [-1, 1] using the
     * transformation t=tanh(x)
     * 
     * @param dFunc
     * @param t
     * @return
     */
    public static double hyperbolicTangentTransform(DoubleUnaryOperator dFunc, double t) {
        if(t == -1 || t == 1) return 0; // integrand value must be 0 at +/- infinity 
        double x = Math.log((1 + t) / (1 - t)) / 2;
        return dFunc.applyAsDouble(x) / (1 - t * t);
    }

    /**
     * Truncates an integral with bounds (-inf, inf) to [-1, 1] using the
     * transformation x = 1/(1-t) - 1/(1+t)
     * 
     * @param dFunc
     * @param t
     * @return
     */
    public static double asymptoticTransform(DoubleUnaryOperator dFunc, double t) {
        if(t == -1 || t == 1) return 0; // integrand value must be 0 at +/- infinity 
        double t_min = 1/(1-t);
        double t_plus = 1/(1+t);
        double x = t_min - t_plus;
        return dFunc.applyAsDouble(x) * (t_min*t_min + t_plus*t_plus);
    }

    
    /**
     * Use the transformation x = (1/t - 1)^(3/2) to convert an integral from the
     * bounds [0, Infinity) to [0, 1]. This is to increase the accuracy of 
     * integration of integrals similar to the hurwitz-zeta function with s=3/2;
     * 
     * @param func the function being integrated over.
     * @param t    the evaluation point on the bounds [0, 1]
     * @return the transformed value at the given point
     */
    public static double hurwitzThreeHalfsTransform(DoubleUnaryOperator func, double t) {
        final double temp = 1 / t - 1;

        // if x is effectively infinite, then the provided function is assumed to have a
        // value of 0 due to implicit convergence requirement
        if (Double.isInfinite(temp) || temp == 0) {
            return 0;
        }

        final double transformedIntegral = func.applyAsDouble(Math.pow(temp, 3d / 2)) * Math.sqrt(temp) / (t * t);

        if (!Double.isFinite(transformedIntegral)) {
            System.out.println();
        }
        assert Double.isFinite(transformedIntegral);
        return 3 * transformedIntegral / 2;
    }

    public static double expInvLogTransform(DoubleUnaryOperator func, double t) {
        double abs_t = Math.abs(t);
        if(abs_t <= 1E-15 || (abs_t - 1) <= 0.03){
            return 0;
        }
        double absLog = Math.log(abs_t);
        double x = Math.exp(t/absLog);
        if(x == 0) return 0;
        double result = func.applyAsDouble(x) * x * (absLog - 1)/(absLog*absLog);
        assert Double.isFinite(result); 
        return result;
    }
}
