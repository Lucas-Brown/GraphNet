package com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators;

import java.util.ArrayList;
import java.util.function.Supplier;

import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.CombinatorMissalignmentException;

public abstract class DirectProbabilityCombinator implements IProbabilityCombinator{

    protected Supplier<IFilter> filterSupplier; 

    public DirectProbabilityCombinator(Supplier<IFilter> filterSupplier){
        this.filterSupplier = filterSupplier;
    }

    public void assignTransferProbabilities(ArrayList<Signal> signals, int key) {
        IFilter[] filters = getFilters(key);
        if(filters.length != signals.size()){
            throw new CombinatorMissalignmentException("Filters do not fit the number of incoming signals.");
        }
        for (int i = 0; i < filters.length; i++) {
            Signal signal = signals.get(i);
            signal.transferProbability = filters[i].getChanceToSend(signal.getOutputStrength());
        }
    }

}
