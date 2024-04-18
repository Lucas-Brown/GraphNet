package com.lucasbrown.NetworkTraining.ApproximationTools;

public class WeightedAverage {
    
    private double prodSum;
    private double weightSum;

    public WeightedAverage(){
        reset();
    }

    public void add(double value, double weight){
        prodSum += value * weight;
        weightSum += weight;
    }

    public double getAverage(){
        return prodSum / weightSum;
    }

    public boolean nonZero(){
        return weightSum != 0;
    }

    public void reset(){
        prodSum = 0;
        weightSum = 0;
    }
}
