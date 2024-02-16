package com.lucasbrown.GraphNetwork.Local;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class BellCurveDistributionAdjuster2Test {
    
    private static final double TOLLERANCE = 1E-4;

    @Test
    public void testLogLikelihoodOfDistribution() 
    {
        BellCurveDistribution parent = new BellCurveDistribution(0, 1);
        BellCurveDistributionAdjuster2 adjuster = new BellCurveDistributionAdjuster2(parent);

        assertEquals(-0.578817117071, adjuster.logLikelihoodOfDistribution(parent, 0, 1), TOLLERANCE);
        assertEquals(-0.675316013282, adjuster.logLikelihoodOfDistribution(parent, 0.5, 1), TOLLERANCE);
        assertEquals(-0.468267130371, adjuster.logLikelihoodOfDistribution(parent, 0, 1.5), TOLLERANCE);

        BellCurveDistribution test1 = new BellCurveDistribution(0.2, 1);
        BellCurveDistribution test2 = new BellCurveDistribution(-0.3, 0.9);
        BellCurveDistribution test3 = new BellCurveDistribution(4, 0.5);

        assertEquals(-0.596396714944, adjuster.logLikelihoodOfDistribution(test1, 0, 1), TOLLERANCE);
        assertEquals(-0.749352322939, adjuster.logLikelihoodOfDistribution(test2, 0.5, 1.2), TOLLERANCE);
        assertEquals(-0.00525425760159, adjuster.logLikelihoodOfDistribution(test3, -2, 1.5), TOLLERANCE);
    }

    @Test
    public void testReinforcePoint()
    {
        BellCurveDistribution parent = new BellCurveDistribution(1.02, 0.34, 100);
        BellCurveDistributionAdjuster2 adjuster = new BellCurveDistributionAdjuster2(parent);

        adjuster.addPoint(1.6, true, 1);
        adjuster.applyAdjustments();

        assertEquals(1.36777459545, adjuster.getMean(), TOLLERANCE);
        assertEquals(0.353800107419, adjuster.getVariance(), TOLLERANCE);
    }
}
