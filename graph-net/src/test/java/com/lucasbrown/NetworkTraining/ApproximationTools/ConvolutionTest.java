package com.lucasbrown.NetworkTraining.ApproximationTools;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.junit.Test;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveFilter;
import com.lucasbrown.GraphNetwork.Distributions.Filter;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.Convolution.FilterDistributionConvolution;

public class ConvolutionTest {

    @Test
    public void testSample() {
        double target = 0;

        BellCurveFilter bcd1 = new BellCurveFilter(0, 1);
        BellCurveFilter bcd2 = new BellCurveFilter(1, 1);
        BellCurveFilter bcd3 = new BellCurveFilter(3, 0.1);
        ArrayList<BellCurveFilter> distributions = new ArrayList<>(List.of(bcd1, bcd2, bcd3));

        ArrayList<ActivationFunction> activators = new ArrayList<>(
                List.of(ActivationFunction.SIGNED_QUADRATIC, ActivationFunction.LINEAR,
                        ActivationFunction.SIGNED_QUADRATIC));

        double[] weights = new double[] { 1, 0, -2};

        FilterDistributionConvolution convolution = new FilterDistributionConvolution(distributions, activators, weights);

        double[] sample = convolution.sample(target);
        double reconstruction = IntStream.range(0, sample.length)
                .mapToDouble(i -> weights[i] * activators.get(i).activator(sample[i]))
                .sum();

        assertEquals(target, reconstruction, 1E-8);
    }
}
