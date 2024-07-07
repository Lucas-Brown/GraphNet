package com.lucasbrown.HelperClasses;

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

    public double getProdSum(){
        return prodSum;
    }

    public double getWeightSum(){
        return weightSum;
    }

    public double getAverage(){
        return prodSum / weightSum;
    }

    public boolean hasValues(){
        return weightSum != 0;
    }

    public void reset(){
        prodSum = 0;
        weightSum = 0;
    }
}
