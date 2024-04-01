package com.lucasbrown.NetworkTraining.ApproximationTools.Convolution;

import java.util.function.DoubleUnaryOperator;

public interface IConvolution extends DoubleUnaryOperator{

    public abstract IConvolution convolveWith(IConvolution g);

}
