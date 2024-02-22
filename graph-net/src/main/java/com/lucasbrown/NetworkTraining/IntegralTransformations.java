package com.lucasbrown.NetworkTraining;

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
        double x = Math.log((1 + t) / (1 - t)) / 2;
        return dFunc.applyAsDouble(x) / (1 - t * t);
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
}
