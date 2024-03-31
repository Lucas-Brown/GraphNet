package com.lucasbrown.NetworkTraining.ApproximationTools;

import static org.junit.Assert.assertEquals;

import java.util.ArrayList;
import java.util.List;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import org.junit.Test;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Distributions.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;

public class ConvolutionTest {

    @Test
    public void testSample() {
        double target = 0;

        BellCurveDistribution bcd1 = new BellCurveDistribution(0, 1);
        BellCurveDistribution bcd2 = new BellCurveDistribution(1, 1);
        BellCurveDistribution bcd3 = new BellCurveDistribution(3, 0.1);
        ArrayList<BellCurveDistribution> distributions = new ArrayList<>(List.of(bcd1, bcd2, bcd3));

        ArrayList<ActivationFunction> activators = new ArrayList<>(
                List.of(ActivationFunction.SIGNED_QUADRATIC, ActivationFunction.LINEAR,
                        ActivationFunction.SIGNED_QUADRATIC));

        double[] weights = new double[] { 1, 0, -2};

        Convolution convolution = new Convolution(distributions, activators, weights);

        double[] sample = convolution.sample(target);
        double reconstruction = IntStream.range(0, sample.length)
                .mapToDouble(i -> weights[i] * activators.get(i).activator(sample[i]))
                .sum();

        assertEquals(target, reconstruction, 1E-8);
    }
}
