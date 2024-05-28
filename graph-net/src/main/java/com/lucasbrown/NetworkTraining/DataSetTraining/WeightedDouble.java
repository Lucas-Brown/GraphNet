package com.lucasbrown.NetworkTraining.DataSetTraining;

import java.util.Collection;

public class WeightedDouble {
    public double weight, value;

    public WeightedDouble(double weight, double value) {
        this.weight = weight;
        this.value = value;
    }

    public static double getWeightSum(Collection<WeightedDouble> doubles){
        return doubles.stream().mapToDouble(wd -> wd.weight).sum();
    }

    public static double getWeightedMean(Collection<WeightedDouble> doubles, double N){
        return doubles.stream().mapToDouble(wd -> wd.weight * wd.value).sum() / N;
    }

    public static double getWeightedVariance(Collection<WeightedDouble> doubles, double N, double mean){
        return doubles.stream().mapToDouble(wd -> wd.weight * (mean - wd.value)*(mean - wd.value)).sum() / N;
    }
}
