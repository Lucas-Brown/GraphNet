package com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators;

import java.util.Collection;
import java.util.Iterator;
import java.util.function.Supplier;

import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.CombinatorMissalignmentException;

public abstract class DirectProbabilityCombinator implements IProbabilityCombinator{

    protected Supplier<IFilter> filterSupplier; 

    public DirectProbabilityCombinator(Supplier<IFilter> filterSupplier){
        this.filterSupplier = filterSupplier;
    }

    @Override
    public double[] getTransferProbabilities(Collection<Signal> signals, int key) {
        IFilter[] filters = getFilters(key);
        if(filters.length != signals.size()){
            throw new CombinatorMissalignmentException("Filters do not fit the number of incoming signals.");
        }

        double[] transferProbs = new double[filters.length];
        Iterator<Signal> sIter = signals.iterator();
        for (int i = 0; i < filters.length; i++) {
            Signal signal = sIter.next();
            transferProbs[i] = filters[i].getChanceToSend(signal.getOutputStrength());
        }
        return transferProbs;
    }

}
