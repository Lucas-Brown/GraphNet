package com.lucasbrown.NetworkTraining;

import java.util.function.DoubleUnaryOperator;
import java.util.function.ToDoubleFunction;

import jsat.linear.Vec;
import jsat.math.Function;

/**
 * Workaround class to make a functional interface out of a regular class
 */
public class DoubleFunction implements Function {

    private ToDoubleFunction<double[]> dFunc;

    public DoubleFunction(ToDoubleFunction<double[]> dFunc) {
        this.dFunc = dFunc;
    }

    public DoubleFunction(DoubleUnaryOperator dFunc) {
        this.dFunc = (double[] x_arr) -> dFunc.applyAsDouble(x_arr[0]);
    }

    @Override
    public double f(double... x) {
        return dFunc.applyAsDouble(x);
    }

    @Override
    public double f(Vec x) {
        return f(x.arrayCopy());
    }
}
