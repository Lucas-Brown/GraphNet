package com.lucasbrown.HelperClasses;

import java.security.InvalidParameterException;
import java.util.ArrayList;

import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;

public abstract class WeightedDoubleCollector implements IExpectationAdjuster {
    
    protected ArrayList<WeightedDouble> newPoints = new ArrayList<>();
    
    public void prepareAdjustment(double weight, double value) {
        newPoints.add(new WeightedDouble(weight, value));
    }

    @Override
    public void prepareAdjustment(double weight, double[] newPoint) {
        if(newPoint.length > 1){
            throw new InvalidParameterException("This distribution only accepts a single degree of input. ");
        }
        prepareAdjustment(weight, newPoint[0]);
    }

    @Override
    public void prepareAdjustment(double[] newData) {
        prepareAdjustment(1, newData);
    }

}
