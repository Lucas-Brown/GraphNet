package com.lucasbrown.NetworkTraining.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.TrainableCombinator;
import com.lucasbrown.HelperClasses.IterableTools;
import com.lucasbrown.HelperClasses.Structs.Pair;
import com.lucasbrown.NetworkTraining.UntrainableNetworkException;

public class WeightsLinearizer {
    
    public final HashMap<INode, Pair<Integer, TrainableCombinator>> vectorNodeOffset;
    public final HashSet<INode> allNodes;
    public final ArrayList<OutputNode> outputNodes; 
    public final int totalNumOfVariables;

    public WeightsLinearizer(GraphNetwork network){
        ArrayList<INode> nodes = network.getNodes();
        allNodes = new HashSet<>(network.getNodes());
        outputNodes = network.getOutputNodes();
        vectorNodeOffset = new HashMap<>(nodes.size());
        verifyAllTrainable(nodes);
        totalNumOfVariables = InitializeOffsetMap();
    }

    private void verifyAllTrainable(ArrayList<INode> nodes) {
        for (INode node : nodes) {
            if(!(node.getCombinator() instanceof TrainableCombinator)){
                throw new UntrainableNetworkException("Network contains a combinator that does not allow training.");
            } 
        }
    }
    
    private int InitializeOffsetMap() {
        int totalNumOfVariables = 0;
        for (INode node : allNodes) {
            vectorNodeOffset.put(node, new Pair<>(totalNumOfVariables, (TrainableCombinator) node.getCombinator()));
            totalNumOfVariables += node.getCombinator().getNumberOfVariables();
        }
        return totalNumOfVariables;
    } 

    public int getTotalNumberOfVariables()
    {
        return totalNumOfVariables;
    }

    public int getLinearIndexOfWeight(INode node, int key, int weight_index) {
        Pair<Integer, TrainableCombinator> indexCombinPair = vectorNodeOffset.get(node);
        return indexCombinPair.u + indexCombinPair.v.getLinearIndexOfWeight(key, weight_index);
    }

    public int getLinearIndexOfBias(INode node, int key) {
        Pair<Integer, TrainableCombinator> indexCombinPair = vectorNodeOffset.get(node);
        return indexCombinPair.u + indexCombinPair.v.getLinearIndexOfBias(key);
    }

    /**
     * returns the portion of the linearized array corresponding to this node
     * @param node
     * @param allDeltas
     * @return
     */
    public double[] nodeSlice(INode node, double[] allDeltas) {
        Pair<Integer, TrainableCombinator> indexCombinPair = vectorNodeOffset.get(node);
        return IterableTools.slice(allDeltas, (int) indexCombinPair.u, indexCombinPair.v.getNumberOfVariables());
    }
}
