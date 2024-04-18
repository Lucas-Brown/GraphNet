package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

import jsat.utils.Pair;

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

    public List<Outcome> getAllOutcomesOfNode(int node_id){
        return outcomesThroughTime.stream().flatMap(node_map -> node_map.get(node_id).stream()).toList();
    }

    public HashMap<Integer, ArrayList<Outcome>> getOutcomesOfKeyFromNode(int node_id){
        List<Outcome> all_outcomes = getAllOutcomesOfNode(node_id);
        HashMap<Integer, ArrayList<Outcome>> keyMap = new HashMap<>(8, 2);
        for(Outcome out : all_outcomes){
            ArrayList<Outcome> outcomesForKey = keyMap.get(out.binary_string);
            if(outcomesForKey == null){
                outcomesForKey = new ArrayList<>();
                outcomesForKey.add(out);
                keyMap.put(out.binary_string, outcomesForKey);
            }
            else
            {
                outcomesForKey.add(out);
            }
        }
        return keyMap;
    }

    /**
     * Having some fun with the naming
     */
    public void burnHistory(){
        outcomesThroughTime.clear();
    }
    
}
