package com.lucasbrown.GraphNetwork.Local;

import java.util.ArrayList;
import java.util.List;

import com.lucasbrown.NetworkTraining.ApproximationTools.Pair;

public class ErrorCollector {

    // Error for each non-unique time step, and input binary string
    private ArrayList<TimeKey> allErrors = new ArrayList<>();

    public void addError(int timestep, int key, double error) {
        allErrors.add(new TimeKey(timestep, key, error));
    }

    public List<Pair<Integer, Double>> getErrorsAtTime(final int time) {
        return allErrors.stream().filter(tk -> tk.time == time).map(TimeKey::toTimeErrorPair).toList();
    }

    public List<Pair<Integer, Double>> getErrorsForKey(final int key) {
        return allErrors.stream().filter(tk -> tk.key == key).map(TimeKey::toKeyErrorPair).toList();
    }

    private static class TimeKey {

        public int time;
        public int key;
        public double error;

        public TimeKey(int time, int key, double error) {
            this.time = time;
            this.key = key;
            this.error = error;
        }

        public Pair<Integer, Double> toTimeErrorPair() {
            return new Pair<Integer, Double>(time, error);
        }

        public Pair<Integer, Double> toKeyErrorPair() {
            return new Pair<Integer, Double>(key, error);
        }

    }
}
