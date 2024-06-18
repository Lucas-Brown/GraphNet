package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.Objects;
import java.util.stream.Stream;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;

public class History {

    public final GraphNetwork network;

    /**
     * for each time, a map from node id to a list of outcomes
     */
    private ArrayList<HashMap<INode, ArrayList<Outcome>>> outcomesThroughTime;

    public History(GraphNetwork network) {
        this.network = network;
        outcomesThroughTime = new ArrayList<>();
    }

    public void captureState() {
        ArrayList<INode> nodes = network.getActiveNodes();
        HashMap<INode, ArrayList<Outcome>> state = new HashMap<>(nodes.size());
        for (INode node : nodes) {
            ArrayList<Outcome> outcomes = node.getState();
            state.put(node, outcomes);

            
            // assertion to make sure all references to previous nodes are maintained
            for(Outcome outcome: outcomes){
                if(outcome.sourceOutcomes == null){
                    continue;
                }
                for(int i = 0 ; i < outcome.sourceOutcomes.length; i++){
                    boolean containsReference = false;
                    ArrayList<Outcome> sourceOutcomes = getStateOfNode(outcomesThroughTime.size()-1, outcome.sourceNodes[i]);
                    for(Outcome so : sourceOutcomes){
                        containsReference |= outcome.sourceOutcomes[i] == so;
                    }
                    assert containsReference;
                }
            }
        }

        outcomesThroughTime.add(state);

    }

    public HashMap<INode, ArrayList<Outcome>> getStateAtTimestep(int timestep) {
        return outcomesThroughTime.get(timestep);
    }

    public ArrayList<Outcome> getStateOfNode(int timestep, INode node) {
        return outcomesThroughTime.get(timestep).get(node);
    }

    public List<ArrayList<Outcome>> getHistoryOfNode(INode node) {
        return outcomesThroughTime.stream()
                .map(map -> map.get(node))
                .map(list -> list == null ? new ArrayList<Outcome>(0) : list)
                .toList();
    }

    public List<Outcome> getAllOutcomesOfNode(INode node) {
        return outcomesThroughTime.stream().flatMap(node_map -> node_map.get(node).stream()).toList();
    }

    public HashMap<Integer, ArrayList<Outcome>> getOutcomesOfKeyFromNode(INode node) {
        List<Outcome> all_outcomes = getAllOutcomesOfNode(node);
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

    public Stream<ArrayList<Outcome>> getAnonymousHistoryStream(){
        return outcomesThroughTime.stream().flatMap(map -> map.values().stream());
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
            for (Entry<INode, ArrayList<Outcome>> nodeOutcome : outcomesThroughTime.get(t).entrySet()) {
                sb.append("Node ");
                sb.append(nodeOutcome.getKey().getID());
                sb.append(": ");
                sb.append(nodeOutcome.getValue().stream().sorted(Outcome::descendingProbabilitiesComparator).limit(2).toList().toString());
                sb.append("\n\t");
            }
            sb.append("\n\n");
        }
        return sb.toString();
    }

}
