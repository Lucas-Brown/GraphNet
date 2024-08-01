package com.lucasbrown.NetworkTraining.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.ITrainableValueCombinator;
import com.lucasbrown.HelperClasses.IterableTools;
import com.lucasbrown.HelperClasses.Structs.Pair;
import com.lucasbrown.NetworkTraining.UntrainableNetworkException;

public class WeightsLinearizer {

    // pairing the offset and the value combinator is just convenient
    // really these are two separate ideas, but they're only ever used together
    public final HashMap<INode, Pair<Integer, ITrainableValueCombinator>> vectorNodeOffset;
    public final HashSet<INode> allNodes;
    public final ArrayList<OutputNode> outputNodes;
    public final int totalNumOfVariables;

    public WeightsLinearizer(GraphNetwork network) {
        ArrayList<INode> nodes = network.getNodes();
        allNodes = new HashSet<>(network.getNodes());
        outputNodes = network.getOutputNodes();
        vectorNodeOffset = new HashMap<>(nodes.size());
        verifyAllTrainable(nodes);
        totalNumOfVariables = InitializeOffsetMap();
    }

    private void verifyAllTrainable(ArrayList<INode> nodes) {
        for (INode node : nodes) {
            if (!(node.getValueCombinator() instanceof ITrainableValueCombinator)) {
                throw new UntrainableNetworkException("Network contains a combinator that does not allow training.");
            }
        }
    }

    private int InitializeOffsetMap() {
        int totalNumOfVariables = 0;
        for (INode node : allNodes) {
            ITrainableValueCombinator comb = (ITrainableValueCombinator) node.getValueCombinator();
            vectorNodeOffset.put(node, new Pair<>(totalNumOfVariables, comb));
            totalNumOfVariables += comb.getNumberOfVariables();
        }
        return totalNumOfVariables;
    }

    public int getTotalNumberOfVariables() {
        return totalNumOfVariables;
    }

    public int getLinearIndexOfWeight(INode node, int key, int weight_index) {
        Pair<Integer, ITrainableValueCombinator> indexCombinPair = vectorNodeOffset.get(node);
        return indexCombinPair.u + indexCombinPair.v.getLinearIndexOfWeight(key, weight_index);
    }

    public int getLinearIndexOfBias(INode node, int key) {
        Pair<Integer, ITrainableValueCombinator> indexCombinPair = vectorNodeOffset.get(node);
        return indexCombinPair.u + indexCombinPair.v.getLinearIndexOfBias(key);
    }

    public double[] getAllParameters(){
        double[] params = new double[totalNumOfVariables];

        for(Pair<Integer, ITrainableValueCombinator> positionPair : vectorNodeOffset.values()){
            int idx = positionPair.u;
            ITrainableValueCombinator valueCombinator = positionPair.v;
            System.arraycopy(valueCombinator.getLinearizedVariables(), 0, params, idx, valueCombinator.getNumberOfVariables());
        }

        return params;
    }

    
    public void setParameter(int i, double value){
        double[] params = new double[totalNumOfVariables];

        for(Pair<Integer, ITrainableValueCombinator> positionPair : vectorNodeOffset.values()){
            int idx = positionPair.u;
            ITrainableValueCombinator valueCombinator = positionPair.v;
            int varNum = valueCombinator.getNumberOfVariables();
            if(i >= idx && i < idx + varNum){
                valueCombinator.setLinearizedVariable(i-idx, value);
                return;
            }
        }
    }

    /**
     * returns the portion of the linearized array corresponding to this node
     * 
     * @param node
     * @param allDeltas
     * @return
     */
    public double[] nodeSlice(INode node, double[] allDeltas) {
        Pair<Integer, ITrainableValueCombinator> indexCombinPair = vectorNodeOffset.get(node);
        return IterableTools.slice(allDeltas, (int) indexCombinPair.u, indexCombinPair.v.getNumberOfVariables());
    }
}
