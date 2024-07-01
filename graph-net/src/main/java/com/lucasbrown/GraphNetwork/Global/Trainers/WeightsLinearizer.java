package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.IterableTools;

public class WeightsLinearizer {
    
    public final HashMap<ITrainable, Integer> vectorNodeOffset;
    public final HashSet<ITrainable> allNodes;
    public final ArrayList<OutputNode> outputNodes; 
    public final int totalNumOfVariables;

    public WeightsLinearizer(GraphNetwork network){
        ArrayList<INode> nodes = network.getNodes();
        allNodes = new HashSet<>(nodes.size());
        outputNodes = network.getOutputNodes();
        vectorNodeOffset = new HashMap<>(nodes.size());
        castAllToTrainable(nodes);
        totalNumOfVariables = InitializeOffsetMap();
    }

    private void castAllToTrainable(ArrayList<INode> nodes) {
        for (INode node : nodes) {
            allNodes.add((ITrainable) node);
        }
    }
    
    private int InitializeOffsetMap() {
        int totalNumOfVariables = 0;
        for (ITrainable node : allNodes) {
            vectorNodeOffset.put(node, totalNumOfVariables);
            totalNumOfVariables += node.getNumberOfVariables();
        }
        return totalNumOfVariables;
    } 

    public int getTotalNumberOfVariables()
    {
        return totalNumOfVariables;
    }

    public int getLinearIndexOfWeight(ITrainable node, int key, int weight_index) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfWeight(key, weight_index);
    }

    public int getLinearIndexOfBias(ITrainable node, int key) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfBias(key);
    }

    /**
     * returns the portion of the linearized array corresponding to this node
     * @param node
     * @param allDeltas
     * @return
     */
    public double[] nodeSlice(ITrainable node, double[] allDeltas) {
        return IterableTools.slice(allDeltas, (int) vectorNodeOffset.get(node), node.getNumberOfVariables());
    }
}
