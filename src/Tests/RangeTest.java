package src.Tests;

import static org.junit.Assert.assertEquals;

import org.junit.Test;

import src.NetworkTraining.LinearRange;

public class RangeTest {

    private final double tollerance = 1E-6;
    
    @Test
    public void testLinearRangeInclusiveInclusive()
    {
        LinearRange range = new LinearRange(-2, 5, 5, true, true);
        assertEquals(-2, range.getValue(0), tollerance);
        assertEquals(1.5, range.getValue(2), tollerance);
        assertEquals(5, range.getValue(4), tollerance);
    }

    @Test
    public void testLinearRangeInclusiveExclusive()
    {
        LinearRange range = new LinearRange(-1, 5, 5, true, false);
        assertEquals(-1, range.getValue(0), tollerance);
        assertEquals(1.4, range.getValue(2), tollerance);
        assertEquals(3.8, range.getValue(4), tollerance);
    }
    
    @Test
    public void testLinearRangeExclusiveInclusive()
    {
        LinearRange range = new LinearRange(-1, 5, 5, false, true);
        assertEquals(0.2,range.getValue(0),  tollerance);
        assertEquals(2.6,range.getValue(2),  tollerance);
        assertEquals(5, range.getValue(4),  tollerance);
    }
    
    @Test
    public void testLinearRangeExclusiveExclusive()
    {
        LinearRange range = new LinearRange(-1, 5, 5, false, false);
        assertEquals(0,range.getValue(0),  tollerance);
        assertEquals(2, range.getValue(2), tollerance);
        assertEquals(4, range.getValue(4), tollerance);
    }

    @Test
    public void testLinearRangeResidue()
    {
        LinearRange range = new LinearRange(0, 1, 5, true, true);
        assertEquals(0, range.getIndexResidualWeight(0.25),  tollerance);
        assertEquals(0.8, range.getIndexResidualWeight(0.3), tollerance);
        assertEquals(0, range.getIndexResidualWeight(1), tollerance);
    }
    
    @Test
    public void testLinearRangeIndex()
    {
        LinearRange range = new LinearRange(0, 1, 5, true, true);
        assertEquals(0, range.getNearestIndex(0.25));
        assertEquals(1, range.getNearestIndex(0.3));
        assertEquals(3, range.getNearestIndex(1));
    }

}
