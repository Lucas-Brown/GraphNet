package com.lucasbrown.NetworkTraining;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.function.DoubleUnaryOperator;

import org.junit.Test;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Distributions.NormalDistribution;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.FilterDistributionConvolution;


public class ConvolutionTest {

    private static final double TOLLERANCE = 1E-8;

    @Test
    public void testConvolution() {

        DoubleUnaryOperator d1 = x -> Math.exp(-x*x/2)/Math.sqrt(2 * Math.PI);
        DoubleUnaryOperator d2 = x -> Math.exp(-(x-1)*(x-1)/2)/Math.sqrt(2 * Math.PI); 

        DoubleUnaryOperator convOp = FilterDistributionConvolution.convolution(d1, d2);

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

        FilterDistributionConvolution convolution = new FilterDistributionConvolution(new ArrayList(List.of(d1, d2)), new ArrayList(List.of(a1, a2)), new double[]{1,1});
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

        FilterDistributionConvolution convolution = new FilterDistributionConvolution(new ArrayList(List.of(d1, d2)), new ArrayList(List.of(a1, a2)), new double[]{2,1});
        assertEquals(0.120853388328, convolution.convolve(0), TOLLERANCE);
        assertEquals(0.151640520652, convolution.convolve(1), TOLLERANCE);
        assertEquals(0.0591454337874, convolution.convolve(5), TOLLERANCE);
    }

    @Test
    public void testSample() {
        double sample_tollerance = 1E-3;

        // can only verify a very simple case 
        double u1 = -0.6;
        double u2 = 1;
        double s1 = 0.5;
        double s2 = 0.85;
        double z = 2.75;

        double s12 = s1*s1;
        double s22 = s2*s2;

        double expected_mean1 = (s22*u1 + s12*(z - u2))/(s12+s22);
        double expected_mean2 = (s12*u2 + s22*(z - u1))/(s12+s22);
        double expected_variance = Math.sqrt(1/(1/s12 + 1/s22));

        // factor of root 2 because it's not a true bell curve
        NormalDistribution d1 = new NormalDistribution(u1, s1);
        NormalDistribution d2 = new NormalDistribution(u2, s2);

        ActivationFunction a1 = ActivationFunction.LINEAR;
        ActivationFunction a2 = ActivationFunction.LINEAR;

        FilterDistributionConvolution convolution = new FilterDistributionConvolution(new ArrayList(List.of(d1, d2)), new ArrayList(List.of(a1, a2)), new double[]{1,1});
        double[][] samples = convolution.sample(z, (int) (10/sample_tollerance/sample_tollerance));

        double mean_1 = 0;
        double mean_2 = 0;
        for (int i = 0; i < samples.length; i++) {
            mean_1 += samples[i][0];
            mean_2 += samples[i][1];
        }

        mean_1/= samples.length;
        mean_2/= samples.length;
        
        double variance_1 = 0;
        double variance_2 = 0;
        for (int i = 0; i < samples.length; i++) {
            variance_1 += (mean_1 - samples[i][0])*(mean_1 - samples[i][0]);
            variance_2 += (mean_2 - samples[i][1])*(mean_2 - samples[i][1]);
        }

        variance_1 = Math.sqrt(variance_1/samples.length);
        variance_2 = Math.sqrt(variance_2/samples.length);

        assertEquals(expected_mean1, mean_1, sample_tollerance);
        assertEquals(expected_mean2, mean_2, sample_tollerance);
        assertEquals(variance_1, variance_2, 1E-8);
        assertEquals(expected_variance, variance_1, sample_tollerance);

    }
}
