package com.lucasbrown.NetworkTraining.ApproximationTools.Convolution;

import java.util.function.DoubleUnaryOperator;

import com.lucasbrown.NetworkTraining.ApproximationTools.DoubleFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.IntegralTransformations;

import jsat.math.integration.Romberg;

public class GenericConvolution implements IConvolution {

    protected DoubleUnaryOperator func;

    public GenericConvolution(DoubleUnaryOperator func){
        this.func = func;
    }
    
    @Override
    public double applyAsDouble(double operand) {
        return func.applyAsDouble(operand);
    }

    @Override
    public IConvolution convolveWith(IConvolution g) {
        return new GenericConvolution(convolve(func, g));
    }

    public static DoubleUnaryOperator convolve(DoubleUnaryOperator func1, DoubleUnaryOperator func2)
    {
        return z -> Romberg.romb(new DoubleFunction(convolutionIntegrand(func1, func2, z)), -1, 1);
    }

    public static DoubleUnaryOperator convolutionIntegrand(DoubleUnaryOperator func1, DoubleUnaryOperator func2, double z) {
        return t -> IntegralTransformations
                .asymptoticTransform(x -> func1.applyAsDouble(x) * func2.applyAsDouble(z - x), t);
    }

}
