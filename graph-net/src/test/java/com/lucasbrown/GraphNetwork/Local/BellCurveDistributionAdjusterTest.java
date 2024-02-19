package com.lucasbrown.GraphNetwork.Local;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

public class BellCurveDistributionAdjusterTest {

    private static final double TOLLERANCE = 1E-4;

    @Test
    public void testLogLikelihoodOfDistribution() {
        BellCurveDistribution parent = new BellCurveDistribution(0, 1);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent, false);

        assertEquals(-1.81822781821, adjuster.logLikelihoodOfDistribution(parent, 0, 1), TOLLERANCE);
        assertEquals(-2.127186179, adjuster.logLikelihoodOfDistribution(parent, 0.5, 1), TOLLERANCE);
        assertEquals(-2.01450384189, adjuster.logLikelihoodOfDistribution(parent, 0, 1.5), TOLLERANCE);

        BellCurveDistribution test1 = new BellCurveDistribution(0.2, 1);
        BellCurveDistribution test2 = new BellCurveDistribution(-0.3, 0.9);
        BellCurveDistribution test3 = new BellCurveDistribution(4, 0.5);

        assertEquals(-1.86999254089, adjuster.logLikelihoodOfDistribution(test1, 0, 1), TOLLERANCE);
        assertEquals(-2.40010541606, adjuster.logLikelihoodOfDistribution(test2, 0.5, 1.2), TOLLERANCE);
        assertEquals(-15.0568125631, adjuster.logLikelihoodOfDistribution(test3, -2, 1.5), TOLLERANCE);
    }

    @Test
    public void testlogLikelihood() {
        BellCurveDistribution parent = new BellCurveDistribution(0, 1);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent, false);

        assertEquals(-1.9189385332, adjuster.logLikelihood(1, true, 0, 1), TOLLERANCE);
        assertEquals(-2.35169066277, adjuster.logLikelihood(1, false, 0, 1), TOLLERANCE);
    }

    @Test
    public void testReinforcePoint() {
        BellCurveDistribution parent = new BellCurveDistribution(1.00516583438, 0.593675532771, 4);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent, false);

        adjuster.addPoint(3, true, 1);
        adjuster.applyAdjustments();

        // low accuracy due to poor estimated expected value. Didn't want to manuallly
        // do newton's method more
        assertEquals(1.21764651754, adjuster.getMean(), 1E-1);
        assertEquals(1.10724879885, adjuster.getVariance(), 1E-1);
    }

    @Test
    public void testDiminishPoint() {
        BellCurveDistribution parent = new BellCurveDistribution(1.00516583438, 0.593675532771, 4);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent, false);

        adjuster.addPoint(3, false, 1);
        adjuster.applyAdjustments();

        // low accuracy due to poor estimated expected value. Didn't want to manuallly
        // do newton's method more
        assertEquals(1.12534636961, adjuster.getMean(), 1E-1);
        assertEquals(0.868384017728, adjuster.getVariance(), 1E-1);
    }

    @Test
    public void integrationTest() {
        assertEquals(Math.expm1(1), BellCurveDistributionAdjuster.integrate(Math::exp, 0, 1), TOLLERANCE);
    }

    @Test
    public void indefiniteIntegrationTest() {
        assertEquals(1d, BellCurveDistributionAdjuster.infiniteIntegral(x -> Math.exp(-x)), TOLLERANCE);
    }
}
