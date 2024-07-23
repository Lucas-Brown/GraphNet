package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.Collection;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
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

    public static void iterateOverHistory(Double[][] targets, List<? extends INode> outputNodes, NetworkHistory networkHistory, HistoryIteratorFunction histFunc){
        
        // loop over all output nodes at every timestep
        for (int timestep = 0; timestep < targets.length; timestep++) {

            for (int i = 0; i < outputNodes.size(); i++) {
                INode outputNode = outputNodes.get(i);
                ArrayList<Outcome> outcomesAtTime = networkHistory.getStateOfRecord(timestep, outputNode);
                Double target = targets[timestep][i];
                
                histFunc.apply(outputNode, outcomesAtTime, target);
            }

        }
    }


    @FunctionalInterface
    public static interface HistoryIteratorFunction{

        public void apply(INode outputNode, ArrayList<Outcome> outcomesAtTime, Double target);
    }
}
