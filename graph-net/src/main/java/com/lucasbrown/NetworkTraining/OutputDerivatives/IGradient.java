package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.Collection;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;

import jsat.linear.Vec;

public interface IGradient {

    public Vec computeGradient(NetworkHistory networkHistory);

    public double getTotalError(NetworkHistory networkHistory);

    public void setTargets(Double[][] targets); 
    public Double[][] getTargets();

    static double getProbabilityVolume(Outcome[] outcomes) {
        return Stream.of(outcomes).mapToDouble(outcome -> outcome.probability).sum();
    }

    static double getProbabilityVolume(Collection<Outcome> outcomes) {
        return outcomes.stream().mapToDouble(outcome -> outcome.probability).sum();
    }
}
