package com.lucasbrown.NetworkTraining.Trainers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ProbabilityCombinators.IProbabilityCombinator;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ITrainableValueCombinator;
import com.lucasbrown.HelperClasses.IterableTools;
import com.lucasbrown.HelperClasses.Structs.Pair;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class FilterLinearizer {
    
    public final HashMap<IFilter, Integer> vectorFilterOffset;
    public final HashSet<IFilter> allFilters;
    public final int totalNumOfVariables;

    public FilterLinearizer(GraphNetwork network){
        ArrayList<INode> nodes = network.getNodes();
        int n_size = nodes.size();
        allFilters = new HashSet<>(n_size);
        vectorFilterOffset = new HashMap<>(n_size);
        collectFilters(nodes);
        totalNumOfVariables = InitializeOffsetMap();
    }

    private void collectFilters(ArrayList<INode> nodes) {
        for (INode node : nodes) {
            IProbabilityCombinator comb = node.getProbabilityCombinator();
            allFilters.addAll(List.of(comb.getAllFilters()));
        }
    }


    private int InitializeOffsetMap() {
        int totalNumOfVariables = 0;
        for (IFilter filter : allFilters) {
            vectorFilterOffset.put(filter, totalNumOfVariables);
            totalNumOfVariables += filter.getNumberOfAdjustableParameters();
        }

        return totalNumOfVariables;
    } 

    public int getTotalNumberOfVariables()
    {
        return totalNumOfVariables;
    }

    
    /**
     * returns the portion of the linearized array corresponding to this filter
     * @param node
     * @param allDeltas
     * @return
     */
    public double[] filterSlice(IFilter filter, double[] allDeltas) {
        return IterableTools.slice(allDeltas, vectorFilterOffset.get(filter), filter.getNumberOfAdjustableParameters());
    }

    public Vec paramsToVector(IFilter filter, double[] filter_derivative) {
        Vec vec = new DenseVector(totalNumOfVariables);
        int start = vectorFilterOffset.get(filter);
        for (int i = 0; i < filter_derivative.length; i++) {
            vec.set(i+start, filter_derivative[i]);
        }
        return vec;
    }

    public Vec addToVector(IFilter filter, double[] filter_derivative, Vec vec) {
        Vec new_vec = new DenseVector(vec);
        int start = vectorFilterOffset.get(filter);
        for (int i = 0; i < filter_derivative.length; i++) {
            int vec_idx = i + start;
            new_vec.set(vec_idx, new_vec.get(vec_idx) + filter_derivative[i]);
        }
        return new_vec;
    }

    
    public double[] getAllParameters(){
        double[] params = new double[totalNumOfVariables];

        for(Entry<IFilter, Integer> entry : vectorFilterOffset.entrySet()){
            IFilter filter = entry.getKey();
            int idx = entry.getValue();
            System.arraycopy(filter.getAdjustableParameters(), 0, params, idx, filter.getNumberOfAdjustableParameters());
        }

        return params;
    }

    public void setParameter(int i, double value){
        for(Entry<IFilter, Integer> entry : vectorFilterOffset.entrySet()){
            IFilter filter = entry.getKey();
            int idx = entry.getValue();
            int varNum = filter.getNumberOfAdjustableParameters();
            if(i >= idx && i < idx + varNum){
                filter.setAdjustableParameter(i-idx, value);
                return;
            }
        }
    }
}
