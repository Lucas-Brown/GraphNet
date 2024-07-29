package com.lucasbrown.NetworkTraining.History;

import java.util.ArrayList;
import java.util.Iterator;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.IOutputNode;

/**
 * Simple shorthand for how this object is used in maintaining the history of the Graph network through Outcome objects
 */
public class NetworkHistory extends History<Outcome, INode> {

    public NetworkHistory(IStateGenerator<INode> stateGenerator) {
        super(stateGenerator);
    }

    public Iterator<HistoryOutputIteratorStruct> outputIterator(ArrayList<? extends IOutputNode> outputNodes){
        return new HistoryOutputIterator(outputNodes);
    }

    public static class HistoryOutputIteratorStruct{
        public final int timestep;
        public final int outputNodeIndex;
        public final IOutputNode node;
        public final ArrayList<Outcome> outcomes;

        public HistoryOutputIteratorStruct(int timestep, int outputNodeIndex, IOutputNode node, ArrayList<Outcome> outcomes){
            this.timestep = timestep;
            this.outputNodeIndex = outputNodeIndex;
            this.node = node;
            this.outcomes = outcomes;
        }

        public HistoryOutputIteratorStruct(HistoryOutputIteratorStruct histStruct){
            this(histStruct.timestep, histStruct.outputNodeIndex, histStruct.node, histStruct.outcomes);
        }
    }

    public class HistoryOutputIterator implements Iterator<HistoryOutputIteratorStruct>{

        private int timestep = 0;
        private int nodeIndex = 0;
        private ArrayList<? extends IOutputNode> outputNodes;

        public HistoryOutputIterator(ArrayList<? extends IOutputNode> outputNodes){
            this.outputNodes = outputNodes;
        }

        @Override
        public boolean hasNext() {
            return timestep < getNumberOfTimesteps();
        }

        @Override
        public HistoryOutputIteratorStruct next() {
            IOutputNode node = outputNodes.get(nodeIndex);
            HistoryOutputIteratorStruct struct = new HistoryOutputIteratorStruct(timestep, nodeIndex, node, getStateOfRecord(timestep, node));
            if(++nodeIndex >= outputNodes.size()){
                nodeIndex = 0;
                timestep++;
            }
            return struct;
        }

    }

}
