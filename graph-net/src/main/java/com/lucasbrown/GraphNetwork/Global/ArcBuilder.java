package com.lucasbrown.GraphNetwork.Global;

import java.util.function.Supplier;

import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.HelperClasses.FunctionalInterfaces.TriFunction;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.IExpectationAdjuster;
import com.lucasbrown.NetworkTraining.DistributionSolverMethods.ITrainableDistribution;

public class ArcBuilder {

    private final GraphNetwork network;

    private Supplier<IFilter> filterSupplier;
    private FilterAdjusterFunction filterAdjusterSupplier;

    public ArcBuilder(final GraphNetwork network) {
        this.network = network;
    }

    public void setFilterSupplier(Supplier<IFilter> filterSupplier) {
        this.filterSupplier = filterSupplier;
    }

    public void setFilterAdjusterSupplier(FilterAdjusterFunction filterAdjusterSupplier) {
        this.filterAdjusterSupplier = filterAdjusterSupplier;
    }

    public Edge build(ITrainable source, ITrainable dest) {
        IFilter filter = filterSupplier.get();
        return network.addNewConnection(source, dest, filter, filterAdjusterSupplier.apply(filter,
                source.getOutputDistribution(), dest.getSignalChanceDistribution()));
    }

    // Shorter name for the tri-function in this context
    @FunctionalInterface
    public static interface FilterAdjusterFunction
            extends TriFunction<IFilter, ITrainableDistribution, ITrainableDistribution, IExpectationAdjuster> {
    }
}
