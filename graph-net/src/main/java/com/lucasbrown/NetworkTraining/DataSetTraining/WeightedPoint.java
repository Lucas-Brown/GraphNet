package com.lucasbrown.NetworkTraining.DataSetTraining;

public class WeightedPoint<T> {
    public double weight;
    public T value;

    public WeightedPoint(double weight, T value) {
        this.weight = weight;
        this.value = value;
    }

    
    @Override
    public String toString(){
        return "weight: " + Double.toString(weight) + "\tb: " + value.toString();
    }
}
