package com.lucasbrown.NetworkTraining;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import org.junit.Test;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;


public class ConvolutionTest {

    private static final double TOLLERANCE = 1E-4;

    @Test
    public void testConvolution() {

        DoubleUnaryOperator d1 = x -> Math.exp(-x*x/2)/Math.sqrt(2 * Math.PI);
        DoubleUnaryOperator d2 = x -> Math.exp(-(x-1)*(x-1)/2)/Math.sqrt(2 * Math.PI); 

        DoubleUnaryOperator convOp = Convolution.convolution(d1, d2);

        assertEquals(0.219695644734, convOp.applyAsDouble(0), TOLLERANCE);
        assertEquals(0.282094791774, convOp.applyAsDouble(1), TOLLERANCE);
        assertEquals(0.00516674633852, convOp.applyAsDouble(5), TOLLERANCE);
        
    }
    
    @Test
    public void testConvolveLinear() {
        BellCurveDistribution d1 = new BellCurveDistribution(0, 1);
        BellCurveDistribution d2 = new BellCurveDistribution(1, 1);

        ActivationFunction a1 = ActivationFunction.LINEAR;
        ActivationFunction a2 = ActivationFunction.LINEAR;

        Convolution convolution = new Convolution(new ArrayList(List.of(d1, d2)), new ArrayList(List.of(a1, a2)), new double[]{1,1});
        assertEquals(0.241970724519, convolution.convolve(0), TOLLERANCE);
        assertEquals(0.398942280401, convolution.convolve(1), TOLLERANCE);
        assertEquals(0.000133830225765, convolution.convolve(5), TOLLERANCE);
    }

    @Test
    public void testConvolveQuadratic() {
        BellCurveDistribution d1 = new BellCurveDistribution(0, 1);
        BellCurveDistribution d2 = new BellCurveDistribution(1, 1);

        ActivationFunction a1 = ActivationFunction.SIGNED_QUADRATIC;
        ActivationFunction a2 = ActivationFunction.SIGNED_QUADRATIC;

        Convolution convolution = new Convolution(new ArrayList(List.of(d1, d2)), new ArrayList(List.of(a1, a2)), new double[]{2,1});
        assertEquals(0.120853388328, convolution.convolve(0), TOLLERANCE);
        assertEquals(0.151640520652, convolution.convolve(1), TOLLERANCE);
        assertEquals(0.0591454337874, convolution.convolve(5), TOLLERANCE);
    }

    @Test
    public void testSample() {

    }
}
