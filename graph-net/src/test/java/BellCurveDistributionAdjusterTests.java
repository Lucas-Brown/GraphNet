
import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;

import java.util.function.DoubleBinaryOperator;
import java.util.stream.IntStream;

import org.junit.Test;

import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistributionAdjuster;


/**
 * Many of these tests fail simply because the interpolation method for the scale is linear.
 */
public class BellCurveDistributionAdjusterTests{
    
    private final double tollerance = 1E-2;
    private final double[] w_values = new double[]{-2,-2,-2,0,0,0,2,2,2};
    private final double[] eta_values = new double[]{0.5,1,2,0.5,1,2,0.5,1,2};

    private final double[] target_shift = new double[]{5.66762550481, 3.86905106051, 2.6308691075, 0, 0, 0, -5.66762550481, -3.86905106051, -2.6308691075};
    private final double[] target_shift_derivative = new double[]{-1.76776853894, -1.09374817063, -0.922936165, -4.23309652141, -4.63031470931, -5.71572176563, -1.76776853894, -1.09374817063, -0.922936165};
    private final double[] target_scale = new double[]{13.1030195497, 8.28497620655, 5.49247225651, 4.23309652138, 2.31515735467, 1.42893044139, 13.1030195497, 8.28497620655, 5.49247225651};
    private final double[] target_scale_derivative = new double[]{-18.4996814202, -5.15513305282, -1.56046443975, -8.30453647596, -1.78329319129, -0.453952803112, -18.4996814202, -5.15513305282, -1.56046443975};

    private void testAllValues(DoubleBinaryOperator evaluater, double[] targets)
    {
        double[] values = IntStream.range(0, targets.length).mapToDouble(i -> evaluater.applyAsDouble(w_values[i], eta_values[i])).toArray();
        assertArrayEquals(targets, values, tollerance);
    }

    @Test
    public void shiftTest()
    {
        testAllValues(BellCurveDistributionAdjuster::getShiftIntegralValue, target_shift);    
    }
    
    @Test
    public void shiftDerivativeTest()
    {
        testAllValues(BellCurveDistributionAdjuster::getShiftIntegralDerivativeValue, target_shift_derivative);    
    }
    
    @Test
    public void scaleTest()
    {
        testAllValues(BellCurveDistributionAdjuster::getScaleIntegralValue, target_scale);    
    }
    
    @Test
    public void scaleDerivativeTest()
    {
        testAllValues(BellCurveDistributionAdjuster::getScaleIntegralDerivativeValue, target_scale_derivative);    
    }
    
    @Test
    public void shiftDirectTest()
    {
        testAllValues(BellCurveDistributionAdjuster::shiftFunction, target_shift);    
    }

    @Test
    public void scaleDirectTest()
    {
        testAllValues(BellCurveDistributionAdjuster::scaleFunction, target_scale);    
    }

    @Test
    public void shiftDirectTestNoTransform()
    {
        testAllValues(BellCurveDistributionAdjuster::shiftFunctionNoTransform, target_shift);    
    }

    @Test
    public void scaleDirectTestNoTransform()
    {
        testAllValues(BellCurveDistributionAdjuster::scaleFunctionNoTransform, target_scale);    
    }

    @Test
    public void integrationTest()
    {
        assertEquals(Math.expm1(1), BellCurveDistributionAdjuster.integrate(Math::exp, 0, 1), tollerance);
    }

    @Test
    public void indefiniteIntegrationTest()
    {
        assertEquals(1d, BellCurveDistributionAdjuster.infiniteIntegral(x -> Math.exp(-x)), tollerance);
    }

    @Test
    public void solveSimpleCubicTest()
    {
        assertEquals(0.9368, BellCurveDistributionAdjuster.solveSimpleCubic(-3.2, -2.7, 5), 1E-4);
    }

    @Test
    public void testNewtonUpdateReinforcePoint()
    {
        BellCurveDistribution parent = new BellCurveDistribution(0, 1);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent);

        adjuster.addPoint(1, true, 1);
        adjuster.newtonUpdate();

        assertEquals(0.035750626998, adjuster.getMean(), 1E-3);
        assertEquals(1.0226023097, adjuster.getVariance(), 1E-3);
    }

    
    @Test
    public void testNewtonUpdateDiminishPoint()
    {
        BellCurveDistribution parent = new BellCurveDistribution(0, 1);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent);

        adjuster.addPoint(1, false, 1);
        adjuster.newtonUpdate();

        assertEquals(-0.0522688359209, adjuster.getMean(), 1E-3);
        assertEquals(0.964924473682, adjuster.getVariance(), 1E-3);
    }

}
