package com.lucasbrown.NetworkTraining;

/**
 * Creates a range of values between the lower and upper bound such that
 * consecutive points are a constant multiple apart
 * For example, 3 points between 1 and 4 -> [1, 2, 4]. Consecutive numbers are a
 * factor of 2 apart
 */
public class MultiplicitiveRange extends Range {

    private final double lowerBound;
    private final double upperBound;
    private final double growthRate;

    public MultiplicitiveRange(double lowerBound, double upperBound, int n_divisions, boolean isLowerInclusive,
            boolean isUpperInclusive) {
        super(n_divisions);
        LinearRange linearRange = new LinearRange(lowerBound, upperBound, n_divisions, isLowerInclusive,
                isUpperInclusive);

        this.lowerBound = linearRange.lowerBound;
        this.upperBound = linearRange.upperBound;
        growthRate = Math.log(this.upperBound / this.lowerBound) / (this.upperBound - this.lowerBound);

        values = linearRange.stream().map(this::linearToMultiplicitive).toArray();
    }

    private double linearToMultiplicitive(double x) {
        return lowerBound * Math.exp(growthRate * (x - lowerBound));
    }

    private double multiplicitiveToLinear(double x) {
        return lowerBound + Math.log(x / lowerBound) / growthRate;
    }

    @Override
    protected double getFloatingIndex(double x) {
        // find the equivalent floating index of the value
        int n0 = (int) (Math.log(x / lowerBound) * (n_divisions - 1) / (growthRate * (upperBound - lowerBound)));

        // use the proper floating index to find the linearly mapped index
        return n0 + (x - values[n0]) / (values[n0 + 1] - values[n0]);
    }

}
