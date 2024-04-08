package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

public class History {

    public final GraphNetwork network;
    private ArrayList<HashMap<Integer, ArrayList<Outcome>>> outcomesThroughTime;

    public History(GraphNetwork network)
    {
        this.network = network;
        outcomesThroughTime = new ArrayList<>();
    }

    public void captureState()
    {
        ArrayList<INode> nodes = network.getNodes();
        HashMap<Integer, ArrayList<Outcome>> state = new HashMap<>(nodes.size());
        for(INode node : nodes){
            state.put(node.getID(), node.getState());
        }

        outcomesThroughTime.add(state);
    }

    public HashMap<Integer, ArrayList<Outcome>> getStateAtTimestep(int timestep){
        return outcomesThroughTime.get(timestep);
    }

    public ArrayList<Outcome> getStateOfNode(int timestep, int node_id){
        return outcomesThroughTime.get(timestep).get(node_id);
    }

    /**
     * Having some fun with the naming
     */
    public void burnHistory(){
        outcomesThroughTime.clear();
    }
    
}
