package com.lucasbrown.NetworkTraining;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map.Entry;
import java.util.stream.Stream;

/**
 * 
 * T = Outcome
 * V = Irecord 
 */
public class History<T, V extends IStateRecord<T>> {

    public final IStateGenerator<V> stateGenerator;

    /**
     * for each time, a map from record id to a list of outcomes
     */
    private ArrayList<HashMap<V, ArrayList<T>>> outcomesThroughTime;

    public History(IStateGenerator<V> stateGenerator) {
        this.stateGenerator = stateGenerator;
        outcomesThroughTime = new ArrayList<>();
    }

    public void captureState() {
        ArrayList<V> records = stateGenerator.getStateRecords();
        HashMap<V, ArrayList<T>> states = new HashMap<>(records.size());
        for (V key : records) {
            ArrayList<T> state = key.getState();
            states.put(key, state);
        }

        outcomesThroughTime.add(states);

    }

    public int getNumberOfTimesteps(){
        return outcomesThroughTime.size();
    }

    public HashMap<V, ArrayList<T>> getStateAtTimestep(int timestep) {
        return outcomesThroughTime.get(timestep);
    }

    public ArrayList<T> getStateOfRecord(int timestep, V key) {
        return outcomesThroughTime.get(timestep).get(key);
    }

    public List<ArrayList<T>> getHistoryOfRecord(V key) {
        return outcomesThroughTime.stream()
                .map(map -> map.get(key))
                .map(list -> list == null ? new ArrayList<T>(0) : list)
                .toList();
    }

    public List<T> getAllOutcomesOfRecord(V key) {
        return outcomesThroughTime.stream().flatMap(record_map -> record_map.get(key).stream()).toList();
    }

    // public HashMap<Integer, ArrayList<T>> getOutcomesOfKeyFromrecord(V record) {
    //     List<T> all_outcomes = getAllOutcomesOfrecord(record);
    //     HashMap<Integer, ArrayList<Outcome>> keyMap = new HashMap<>(8, 2);
    //     for (T out : all_outcomes) {
    //         ArrayList<T> outcomesForKey = keyMap.get(out.binary_string);
    //         if (outcomesForKey == null) {
    //             outcomesForKey = new ArrayList<>();
    //             outcomesForKey.add(out);
    //             keyMap.put(out.binary_string, outcomesForKey);
    //         } else {
    //             outcomesForKey.add(out);
    //         }
    //     }
    //     return keyMap;
    // }

    public Stream<ArrayList<T>> getAnonymousHistoryStream(){
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
            for (Entry<V, ArrayList<T>> record : outcomesThroughTime.get(t).entrySet()) {
                sb.append("record ");
                sb.append(record.getKey().toString());
                sb.append(": ");
                sb.append(record.getValue().stream().toString());
                sb.append("\n\t");
            }
            sb.append("\n\n");
        }
        return sb.toString();
    }

}
