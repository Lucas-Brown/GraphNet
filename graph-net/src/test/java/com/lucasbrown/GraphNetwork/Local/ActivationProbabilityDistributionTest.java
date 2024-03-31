package com.lucasbrown.GraphNetwork.Local;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;

public class ActivationProbabilityDistributionTest {

    @Test
    public void testGetMeanOfAppliedActivation() {
        BellCurveDistribution dist = new BellCurveDistribution(1, 2);

        assertEquals(1, dist.getMeanOfAppliedActivation(ActivationFunction.LINEAR, 1), 1E-6);
        assertEquals(2.22014110826, dist.getMeanOfAppliedActivation(ActivationFunction.SIGNED_QUADRATIC, 1), 1E-6);
    }

    @Test
    public void testGetVarianceOfAppliedActivation() {
        BellCurveDistribution dist = new BellCurveDistribution(1, 2);

        assertEquals(Math.sqrt(2), dist.getVarianceOfAppliedActivation(ActivationFunction.LINEAR, 1), 1E-6);
        assertEquals(3.51544954565, dist.getVarianceOfAppliedActivation(ActivationFunction.SIGNED_QUADRATIC, 1), 1E-6);
    }
}
