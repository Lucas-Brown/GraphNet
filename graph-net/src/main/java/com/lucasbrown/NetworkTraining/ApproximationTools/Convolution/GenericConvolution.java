package com.lucasbrown.NetworkTraining.ApproximationTools.Convolution;

import java.util.function.DoubleUnaryOperator;

import com.lucasbrown.NetworkTraining.ApproximationTools.DoubleFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.IntegralTransformations;

import jsat.math.integration.Romberg;

/**
 * This class represents a generic convolution operation. It implements the IConvolution interface.
 */
public class GenericConvolution implements IConvolution {

    /**
     * The function to be used for the convolution. It should take one double argument and return a double value.
     */
    protected DoubleUnaryOperator func;

    /**
     * Constructor that takes in a DoubleUnaryOperator as an argument, which will be used for the convolution.
     *
     * @param func A DoubleUnaryOperator representing the function to be convolved with another function.
     */
    public GenericConvolution(DoubleUnaryOperator func){
        this.func = func;
    }
    
    /**
     * Applies the unary operator on the operand (a single input).
     *
     * @param operand An operand of type double.
     * @return Returns the result after applying the function represented by 'func' on the given operand.
     */
    @Override
    public double applyAsDouble(double operand) {
        return func.applyAsDouble(operand);
    }

    /**
     * Convolves two functions together using Romberg's method for numerical integration.
     *
     * @param g Another instance implementing IConvolution, representing the second function to be convolved with.
     * @return A new instance of GenericConvolution representing the resulting function from convolving 'this' with 'g'.
     */
    @Override
    public IConvolution convolveWith(IConvolution g) {
        return new GenericConvolution(convolve(func, g));
    }

    /**
     * Performs the actual convolution between two functions.
     *
     * @param func1 One of the functions being convolved.
     * @param func2 The other function being convolved.
     * @return A DoubleUnaryOperator representing the resultant function obtained through convolution.
     */
    public static DoubleUnaryOperator convolve(DoubleUnaryOperator func1, DoubleUnaryOperator func2)
    {
        // Perform the convolution integral over -1 <= t <= 1
        return z -> Romberg.romb(new DoubleFunction(convolutionIntegrand(func1, func2, z)), -1, 1);
    }

    /**
     * Defines the integrand of the convolution integral.
     *
     * @param func1 First function involved in the convolution.
     * @param func2 Second function involved in the convolution.
     * @param z Value at which we want to evaluate the convolution.
     * @return A DoubleUnaryOperator representing the integrand of the convolution integral.
     */
    public static DoubleUnaryOperator convolutionIntegrand(DoubleUnaryOperator func1, DoubleUnaryOperator func2, double z) {
        // Apply asymptotic transformation to improve accuracy near singularities
        return t -> IntegralTransformations.asymptoticTransform(x -> func1.applyAsDouble(x) * func2.applyAsDouble(z - x), t);
    }

}
