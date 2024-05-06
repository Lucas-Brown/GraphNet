package com.lucasbrown.NetworkTraining.DataSetTraining;

public class WeightedPoint<T> {
    public double weight;
    public T value;

    public WeightedPoint(double weight, T value) {
        this.weight = weight;
        this.value = value;
    }
}
