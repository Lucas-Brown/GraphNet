package src.NetworkTraining;

import java.util.Arrays;
import java.util.Iterator;
import java.util.stream.DoubleStream;

public abstract class Range implements Iterable<double[]>{
    
    protected double[] values;

    /**
     * Check whether the provided value is within the range
     * @param x
     * @return
     */
    public boolean isWithinRange(double x)
    {
        return x >= values[0] && x <= values[values.length - 1];
    }

    public int getNumberOfPoints()
    {
        return values.length;
    }

    /**
     * Whether the index is within the range
     * @param index
     * @return
     */
    public boolean isValidIndex(int index)
    {
        return index >= 0 && index < values.length;
    }

    /**
     * Get the value at the index
     * @param index
     * @return
     */
    public double getValue(int index)
    {
        return values[index];
    }

    public Iterator<double[]> iterator()
    {
        return Arrays.asList(values).iterator();
    }

    public DoubleStream stream()
    {
        return Arrays.stream(values);
    }

    /**
     * Get the index of the lower 
     * @param x
     * @return
     */
    public abstract int getNearestIndex(double x);
    
    /**
     * Returns how close x is to the nearest value less than x
     * Different range mappings may choose to weigh how close a value is to the next nearest index.
     * For example, let x = 1.5 with it's nearest neighbors 1 and 2.
     * a linear mapping would return 0.5
     * a non linear mapping (such as 2^x -1) may return 0.41
     * @param x
     * @return
     */
    public abstract double getIndexResidualWeight(double x);

}
