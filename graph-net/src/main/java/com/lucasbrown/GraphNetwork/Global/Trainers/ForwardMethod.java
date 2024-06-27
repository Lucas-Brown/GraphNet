package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;

public abstract class ForwardMethod implements IGradient{
    
    private GraphNetwork network;

    protected int totalNumOfVariables;
    protected HashMap<ITrainable, Integer> vectorNodeOffset;
    protected HashSet<ITrainable> allNodes;

    public ForwardMethod(GraphNetwork network){
        this.network = network;
        castAllToTrainable();
        InitializeOffsetMap();
    }
    
    private void InitializeOffsetMap() {
        vectorNodeOffset = new HashMap<>(allNodes.size());

        totalNumOfVariables = 0;
        for (ITrainable node : allNodes) {
            vectorNodeOffset.put(node, totalNumOfVariables);
            totalNumOfVariables += node.getNumberOfVariables();
        }
    }

    private void castAllToTrainable() {
        ArrayList<INode> nodes = network.getNodes();
        allNodes = new HashSet<>(nodes.size());
        for (INode node : nodes) {
            allNodes.add((ITrainable) node);
        }
    }

    protected int getLinearIndexOfWeight(ITrainable node, int key, int weight_index) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfWeight(key, weight_index);
    }

    protected int getLinearIndexOfBias(ITrainable node, int key) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfBias(key);
    }
}
