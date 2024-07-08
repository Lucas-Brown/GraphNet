package com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators;

import java.util.Arrays;
import java.util.function.Supplier;

import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.HelperClasses.IterableTools;

public class SimpleProbabilityCombinator extends DirectProbabilityCombinator {
 
    private IFilter[] filters;

    public SimpleProbabilityCombinator(Supplier<IFilter> filterSupplier){
        super(filterSupplier);
        filters = new IFilter[0];
    }

    @Override
    public void notifyNewIncomingConnection() {
        // add another slot
        if(filters.length == 0){
            filters = new IFilter[]{filterSupplier.get()};
        }
        else{
            IFilter[] newArr = new IFilter[2*filters.length];
            System.arraycopy(filters, 0, newArr, 0, filters.length);
            for(int i = filters.length; i<newArr.length; i++){
                newArr[i] = filterSupplier.get();
            }
            filters = newArr;
        }
    }

    @Override
    public IFilter[] getFilters(int key) {
        return IterableTools.applyMask(filters, key);
    }

    @Override
    public IFilter[] getAllFilters() {
        return Arrays.copyOf(filters, filters.length);
    }

}
