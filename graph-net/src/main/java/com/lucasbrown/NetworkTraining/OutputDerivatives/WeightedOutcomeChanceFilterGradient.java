package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.INetworkGradient;
import com.lucasbrown.NetworkTraining.OutputDerivatives.ErrorFunction.CrossEntropy;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class WeightedOutcomeChanceFilterGradient implements IGradient {

    protected Double[][] targets;
    protected INetworkGradient networkGradientEvaluater;
    protected ArrayList<OutputNode> outputNodes;
    protected int totalNumOfVariables;

    private static double temperature = 0.1;
    private ErrorFunction errorFunction;
    private static final CrossEntropy crossEntropy = new CrossEntropy();

    private NetworkHistory networkHistory;
    ArrayList<HashMap<Outcome, Vec>> networkGradient;

    public WeightedOutcomeChanceFilterGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater,
            Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables) {
        this.targets = targets;
        this.networkGradientEvaluater = networkGradientEvaluater;
        this.totalNumOfVariables = totalNumOfVariables;
        this.errorFunction = errorFunction;

        outputNodes = network.getOutputNodes();
    }

    @Override
    public Vec computeGradient(NetworkHistory networkHistory) {
        this.networkHistory = networkHistory;
        networkGradient = networkGradientEvaluater.getGradient(networkHistory);

        Vec probGrad = gradientOfTargets();
        Vec valueGrad = gradientOfValues(networkHistory, networkGradient);
        double probError = errorOfTargets();
        double valueError = errorOfValues();

        return probGrad.add(valueGrad);

    }

    private Vec gradientOfTargets() {

        final int out_size = outputNodes.size();
        Vec[] gradient_prob = new Vec[out_size];
        int[] T_prob = new int[out_size];

        Vec probGrad = new DenseVector(totalNumOfVariables);

        if (out_size == 0) {
            return probGrad;
        }

        for (int i = 0; i < out_size; i++) {
            gradient_prob[i] = new DenseVector(totalNumOfVariables);
        }

        localIterateOverHistory().forEachRemaining(struct -> {
            boolean is_value = struct.target != null;

            // skip when there's no outcome
            if (struct.outcomes == null || struct.outcomes.isEmpty()) {
                return;
            }

            // probabilty component
            gradient_prob[struct.outputNodeIndex]
                    .mutableAdd(gradientOfTarget(struct.outcomes, struct.gradientAtTime, is_value ? 1 : 0));
            T_prob[struct.outputNodeIndex]++;
        });

        // loop over all output nodes at every timestep

        for (int i = 0; i < out_size; i++) {
            if (T_prob[i] == 0)
                continue;

            probGrad.mutableAdd(gradient_prob[i].divide(T_prob[i]));
        }
        return probGrad;
    }

    private Vec gradientOfValues(NetworkHistory networkHistory, ArrayList<HashMap<Outcome, Vec>> networkGradient) {

        final int out_size = outputNodes.size();
        Vec[] gradient_value = new Vec[out_size];
        int[] T_value = new int[out_size];

        Vec valueGrad = new DenseVector(totalNumOfVariables);

        if (out_size == 0) {
            return valueGrad;
        }

        for (int i = 0; i < out_size; i++) {
            gradient_value[i] = new DenseVector(totalNumOfVariables);
        }

        // loop over all output nodes at every timestep
        localIterateOverHistory().forEachRemaining(struct -> {
            boolean is_value = struct.target != null;

            // value component
            if (!is_value) {
                return;
            }

            // skip when there's no outcome
            if (struct.outcomes == null || struct.outcomes.isEmpty()) {
                return;
            }

            Vec gradient_value_at_time = weightedValueGradient(struct.outcomes, struct.gradientAtTime,
                    (double) struct.target);
            gradient_value[struct.outputNodeIndex].mutableAdd(gradient_value_at_time);
            T_value[struct.outputNodeIndex]++;
        });

        for (int i = 0; i < out_size; i++) {
            if (T_value[i] == 0)
                continue;

            valueGrad.mutableAdd(gradient_value[i].divide(T_value[i]));
        }
        return valueGrad.multiply(10);
    }

    private Vec weightedValueGradient(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime,
            double target) {
        Vec gradient = new DenseVector(totalNumOfVariables);

        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);

        if (probabilityVolume == 0) {
            return gradient;
        }

        double totalOutputError = 0;
        for (Outcome outcome : outcomesAtTime) {

            double errorOfOutput = errorFunction.error(outcome.activatedValue, target);
            errorOfOutput = Math.exp(-errorOfOutput / temperature);
            totalOutputError += errorOfOutput;

            Vec networkDerivative = gradientAtTime.get(outcome);
            if (networkDerivative.nnz() == 0) {
                continue;
            }

            assert networkDerivative.countNaNs() == 0;

            // accumulate jacobians
            gradient.mutableAdd(networkDerivative.multiply(errorOfOutput));
        }
        if (totalOutputError == 0) {
            gradient.mutableMultiply(0);
            return gradient;
        }

        assert gradient.countNaNs() == 0;
        gradient.mutableDivide(-totalOutputError * probabilityVolume);
        return gradient;
    }

    private Vec gradientOfTarget(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime,
            double targetProbability) {
        Vec gradient = new DenseVector(totalNumOfVariables);

        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);

        for (Outcome outcome : outcomesAtTime) {
            Vec networkDerivative = gradientAtTime.get(outcome);

            // accumulate jacobians
            gradient.mutableAdd(networkDerivative);
        }
        gradient.mutableMultiply(crossEntropy.error_derivative(probabilityVolume, targetProbability));
        assert gradient.countNaNs() == 0;
        return gradient;
    }

    private double errorOfTargets() {

        final int out_size = outputNodes.size();
        double[] error_prob = new double[out_size];
        int[] T_prob = new int[out_size];

        if (out_size == 0) {
            return 0;
        }

        localIterateOverHistory().forEachRemaining(struct -> {
            boolean is_value = struct.target != null;

            // skip when there's no outcome
            if (struct.outcomes == null || struct.outcomes.isEmpty()) {
                return;
            }

            // probabilty component
            error_prob[struct.outputNodeIndex] += errorOfTarget(struct.outcomes, struct.gradientAtTime,
                    is_value ? 1 : 0);
            T_prob[struct.outputNodeIndex]++;
        });

        // loop over all output nodes at every timestep

        double error = 0;
        for (int i = 0; i < out_size; i++) {
            if (T_prob[i] == 0)
                continue;

            error += error_prob[i] / T_prob[i];
        }
        return error;
    }

    private double errorOfValues() {

        final int out_size = outputNodes.size();
        double[] error_value = new double[out_size];
        int[] T_value = new int[out_size];

        if (out_size == 0) {
            return 0;
        }

        localIterateOverHistory().forEachRemaining(struct -> {
            boolean is_value = struct.target != null;

            // value component
            if (!is_value) {
                return;
            }

            // skip when there's no outcome
            if (struct.outcomes == null || struct.outcomes.isEmpty()) {
                return;
            }

            // probabilty component
            error_value[struct.outputNodeIndex] += errorOfWeightedValue(struct.outcomes, struct.gradientAtTime,
                    (double) struct.target);
            T_value[struct.outputNodeIndex]++;
        });

        // loop over all output nodes at every timestep

        double error = 0;
        for (int i = 0; i < out_size; i++) {
            if (T_value[i] == 0)
                continue;

            error += error_value[i] / T_value[i];
        }
        return error;
    }

    private double errorOfTarget(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime,
            double targetProbability) {
        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);
        return crossEntropy.error(probabilityVolume, targetProbability);
    }

    private double errorOfWeightedValue(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime,
            double target) {

        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);

        if (probabilityVolume == 0) {
            return 0;
        }

        double totalOutputError = 0;
        double error = 0;
        for (Outcome outcome : outcomesAtTime) {

            double errorOfOutput = errorFunction.error(outcome.activatedValue, target);
            errorOfOutput = Math.exp(-errorOfOutput / temperature);
            totalOutputError += errorOfOutput;
            error += outcome.probability * errorOfOutput;
        }
        if (totalOutputError == 0) {
            return 0;
        }
        return error / totalOutputError;
    }

    @Override
    public double getTotalError(NetworkHistory networkHistory) {
        this.networkHistory = networkHistory;
        networkGradient = null;
        return errorOfTargets() + errorOfValues();
    }

    private HistoryGradientIterator localIterateOverHistory() {
        return new HistoryGradientIterator(networkHistory, outputNodes, networkGradient, targets);
    }

    @Override
    public void setTargets(Double[][] targets) {
        this.targets = targets;
    }

    @Override
    public Double[][] getTargets() {
        return targets;
    }
}
