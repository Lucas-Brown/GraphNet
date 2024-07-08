package com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators;

import java.util.Arrays;
import java.util.function.Supplier;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.HelperClasses.IterableTools;


public class ComplexProbabilityCombinator extends DirectProbabilityCombinator {

    private IFilter[][] filters;

    protected int numFilters = 0;

    public ComplexProbabilityCombinator(Supplier<IFilter> filterSupplier){
        super(filterSupplier);
        filters = new IFilter[1][1];
        filters[0] = new IFilter[0];
    }

    @Override
    public void notifyNewIncomingConnection() {
        appendWeightsAndBiases();
    }

    /**
     * Adds another layer of depth to the weights and biases hyper array
     */
    private void appendWeightsAndBiases() {
        final int old_size = filters.length;
        final int new_size = old_size * 2;

        numFilters += numFilters + old_size;

        // the first half doesn't need to be changed
        filters = Arrays.copyOf(filters, new_size);

        // the second half needs entirely new data
        for (int i = old_size; i < new_size; i++) {

            // populate the filter array
            int count = filters[i - old_size].length + 1;
            filters[i] = new IFilter[count];
            for (int j = 0; j < count; j++) {
                filters[i][j] = filterSupplier.get();
            }
        }
    }

    @Override
    public IFilter[] getFilters(int key) {
        IFilter[] fitler = filters[key];
        return Arrays.copyOf(fitler, fitler.length);
    }

    @Override
    public IFilter[] getAllFilters() {
        IFilter[] flat = new IFilter[numFilters];
        IterableTools.flatten(filters, flat);
        return flat;
    }

}
