package com.lucasbrown.GraphNetwork.Global;

import java.util.function.Supplier;

import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.ApproximationTools.TriFunction;
import com.lucasbrown.NetworkTraining.DataSetTraining.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;
import com.lucasbrown.NetworkTraining.DataSetTraining.ITrainableDistribution;

public class ArcBuilder {
    
    private final GraphNetwork network; 

    private Supplier<IFilter> filterSupplier;
    private FilterAdjusterFunction filterAdjusterSupplier;

    public ArcBuilder(final GraphNetwork network){
        this.network = network;
    }
    
    public void setFilterSupplier(Supplier<IFilter> filterSupplier) {
        this.filterSupplier = filterSupplier;
    }

    public void setFilterAdjusterSupplier(FilterAdjusterFunction filterAdjusterSupplier) {
        this.filterAdjusterSupplier = filterAdjusterSupplier;
    }

    public Arc build(INode source, INode dest){
        IFilter filter = filterSupplier.get();
        return network.addNewConnection(source, dest, filter, filterAdjusterSupplier.apply(filter, source.getOutputDistribution(), dest.getSignalChanceDistribution()));
    }

    // Shorter name for the tri-function in this context
    @FunctionalInterface
    public static interface FilterAdjusterFunction extends TriFunction<IFilter, ITrainableDistribution, ITrainableDistribution, IExpectationAdjuster>{}
}
