package src.NetworkTraining;

import java.util.Arrays;
import java.util.stream.DoubleStream;

/**
 * Creates a range of values between the lower and upper bound such that consecutive points are a constant multiple apart
 * For example, 3 points between 1 and 4 -> [1, 2, 4]. Consecutive numbers are a factor of 2 apart
 */
public class MultiplicitiveRange extends Range{

    private final double lowerBound;
    private final double upperBound;
    private final double growthRate;

    public MultiplicitiveRange(double lowerBound, double upperBound, int n_divisions, boolean isLowerInclusive,
            boolean isUpperInclusive) 
    {
        super(n_divisions);
        LinearRange linearRange = new LinearRange(lowerBound, upperBound, n_divisions, isLowerInclusive, isUpperInclusive);

        this.lowerBound = linearRange.lowerBound;
        this.upperBound = linearRange.upperBound;
        growthRate = Math.log(this.upperBound/this.lowerBound)/(this.upperBound-this.lowerBound);

        values = linearRange.stream().map(this::linearToMultiplicitive).toArray();
    }
    
    private double linearToMultiplicitive(double x)
    {
        return lowerBound * Math.exp(growthRate * (x - lowerBound));
    }

    private double multiplicitiveToLinear(double x)
    {
        return lowerBound + Math.log(x/lowerBound)/growthRate;
    }

    @Override
    protected double getFloatingIndex(double x)
    {
        return 1 + Math.log(x)*(n_divisions-1)/(growthRate * (upperBound - lowerBound)); 
    }


}
