package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

public class History {

    public final GraphNetwork network;

    /**
     * for each time, a map from node id to a list of outcomes
     */
    private ArrayList<HashMap<Integer, ArrayList<Outcome>>> outcomesThroughTime;

    public History(GraphNetwork network) {
        this.network = network;
        outcomesThroughTime = new ArrayList<>();
    }

    public void captureState() {
        ArrayList<INode> nodes = network.getActiveNodes();
        HashMap<Integer, ArrayList<Outcome>> state = new HashMap<>(nodes.size());
        for (INode node : nodes) {
            state.put(node.getID(), node.getState());
        }

        outcomesThroughTime.add(state);
    }

    public HashMap<Integer, ArrayList<Outcome>> getStateAtTimestep(int timestep) {
        return outcomesThroughTime.get(timestep);
    }

    public ArrayList<Outcome> getStateOfNode(int timestep, int node_id) {
        return outcomesThroughTime.get(timestep).get(node_id);
    }

    public List<ArrayList<Outcome>> getHistoryOfNode(int node_id) {
        return outcomesThroughTime.stream()
                .map(map -> map.get(node_id))
                .map(list -> list == null ? new ArrayList<Outcome>(0) : list)
                .toList();
    }

    public List<Outcome> getAllOutcomesOfNode(int node_id) {
        return outcomesThroughTime.stream().flatMap(node_map -> node_map.get(node_id).stream()).toList();
    }

    public HashMap<Integer, ArrayList<Outcome>> getOutcomesOfKeyFromNode(int node_id) {
        List<Outcome> all_outcomes = getAllOutcomesOfNode(node_id);
        HashMap<Integer, ArrayList<Outcome>> keyMap = new HashMap<>(8, 2);
        for (Outcome out : all_outcomes) {
            ArrayList<Outcome> outcomesForKey = keyMap.get(out.binary_string);
            if (outcomesForKey == null) {
                outcomesForKey = new ArrayList<>();
                outcomesForKey.add(out);
                keyMap.put(out.binary_string, outcomesForKey);
            } else {
                outcomesForKey.add(out);
            }
        }
        return keyMap;
    }

    /**
     * Having some fun with the naming
     */
    public void burnHistory() {
        outcomesThroughTime.clear();
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder();
        for (int t = 0; t < outcomesThroughTime.size(); t++) {
            sb.append("Time Step ");
            sb.append(t);
            sb.append("\n\t");
            for (Entry<Integer, ArrayList<Outcome>> nodeOutcome : outcomesThroughTime.get(t).entrySet()) {
                sb.append("Node ");
                sb.append(nodeOutcome.getKey());
                sb.append(": ");
                sb.append(nodeOutcome.getValue().toString());
                sb.append("\n\t");
            }
            sb.append("\n\n");
        }
        return sb.toString();
    }

}
