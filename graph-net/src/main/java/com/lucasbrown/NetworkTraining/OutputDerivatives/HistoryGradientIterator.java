package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.IOutputNode;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.History.NetworkHistory.HistoryOutputIterator;
import com.lucasbrown.NetworkTraining.History.NetworkHistory.HistoryOutputIteratorStruct;
import com.lucasbrown.NetworkTraining.OutputDerivatives.HistoryGradientIterator.GradientOutputStruct;

import jsat.linear.Vec;

public class HistoryGradientIterator implements Iterator<GradientOutputStruct>{

    private final HistoryOutputIterator histIter;
    private final ArrayList<HashMap<Outcome, Vec>> networkGradient;
    private Double[][] targets;

    public HistoryGradientIterator(NetworkHistory networkHistory, ArrayList<? extends IOutputNode> outputNodes, ArrayList<HashMap<Outcome, Vec>> networkGradient, Double[][] targets) {
        histIter = networkHistory.new HistoryOutputIterator(outputNodes);
        this.networkGradient = networkGradient;
        this.targets = targets;
    }

    @Override
    public boolean hasNext() {
        return histIter.hasNext();
    }

    @Override
    public GradientOutputStruct next() {
        HistoryOutputIteratorStruct struct = histIter.next();
        return new GradientOutputStruct(struct, networkGradient == null ? null : networkGradient.get(struct.timestep), targets[struct.timestep][struct.outputNodeIndex]);
    }
    
    public static class GradientOutputStruct extends HistoryOutputIteratorStruct{

        public final HashMap<Outcome, Vec> gradientAtTime;
        public final Double target;

        public GradientOutputStruct(int timestep, int outputNodeIndex, IOutputNode node, ArrayList<Outcome> outcomes, HashMap<Outcome, Vec> gradientAtTime, Double target) {
            super(timestep, outputNodeIndex, node, outcomes);
            this.gradientAtTime = gradientAtTime;
            this.target = target;
        }

        public GradientOutputStruct(HistoryOutputIteratorStruct histStruct, HashMap<Outcome, Vec> gradientAtTime, Double target) {
            super(histStruct);
            this.gradientAtTime = gradientAtTime;
            this.target = target;
        }
        
    }
}
