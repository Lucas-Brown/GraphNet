package com.lucasbrown.NetworkTraining.OutputDerivatives;

import com.lucasbrown.NetworkTraining.History.NetworkHistory;

import jsat.linear.Vec;

public interface IGradient {

    public Vec computeGradient(NetworkHistory networkHistory);

    public double getTotalError(NetworkHistory networkHistory);

    public void setTargets(Double[][] targets); 
    public Double[][] getTargets();
}
