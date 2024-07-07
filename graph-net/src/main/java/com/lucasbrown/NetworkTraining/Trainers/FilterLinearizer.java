package com.lucasbrown.NetworkTraining.Trainers;

import java.util.ArrayList;
import java.util.Collection;
import java.util.HashMap;
import java.util.HashSet;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Edge;
import com.lucasbrown.GraphNetwork.Local.Filters.IFilter;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.HelperClasses.IterableTools;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class FilterLinearizer {
    
    public final HashMap<IFilter, Integer> vectorFilterOffset;
    public final HashSet<IFilter> allFilters;
    public final int totalNumOfVariables;

    public FilterLinearizer(GraphNetwork network){
        ArrayList<INode> nodes = network.getNodes();
        allFilters = new HashSet<>(nodes.size());
        vectorFilterOffset = new HashMap<>(nodes.size());
        collectFilters(nodes);
        totalNumOfVariables = InitializeOffsetMap();
    }

    private void collectFilters(ArrayList<INode> nodes) {
        for (INode node : nodes) {
            Collection<Edge> arcs = node.getAllIncomingConnections();
            allFilters.addAll(arcs.stream().map(Edge::getFilter).toList());
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
}
