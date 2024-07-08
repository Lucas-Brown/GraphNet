package com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators;

import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Local.Signal;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;

public interface IProbabilityCombinator {
    
    /**
     * Notify the combinator that the node has a new incoming connection
     */
    public void notifyNewIncomingConnection();

    public IFilter[] getFilters(int key);

    public IFilter[] getAllFilters();

    /**
     * Sets the transfer probability of each signal based on its activated strength. 
     * @param signals 
     * @param key
     */
    public void assignTransferProbabilities(ArrayList<Signal> signals, int key);

}
