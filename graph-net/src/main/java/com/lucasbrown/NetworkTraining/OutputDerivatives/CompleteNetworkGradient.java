package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.INetworkGradient;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class CompleteNetworkGradient implements IGradient {

    private ErrorFunction errorFunction;
    private Double[][] targets;
    private INetworkGradient networkGradientEvaluater;
    private INetworkGradient probabilityGradientEvaluater;
    private Vec gradient;
    protected ArrayList<OutputNode> outputNodes;
    protected int totalNumOfVariables;

    public CompleteNetworkGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater,
            INetworkGradient probabilityGradientEvaluater, ErrorFunction errorFunction, Double[][] targets,
            int totalNumOfVariables) {
        this.targets = targets;
        this.networkGradientEvaluater = networkGradientEvaluater;
        this.probabilityGradientEvaluater  = probabilityGradientEvaluater;
        this.errorFunction = errorFunction;
        this.totalNumOfVariables = totalNumOfVariables;
        outputNodes = network.getOutputNodes();
    }

    public Vec computeGradient(NetworkHistory networkHistory) {
        ArrayList<HashMap<Outcome, Vec>> networkGradient = networkGradientEvaluater.getGradient(networkHistory);
        ArrayList<HashMap<Outcome, Vec>> probabilityGradient = probabilityGradientEvaluater.getGradient(networkHistory);
        gradient = new DenseVector(totalNumOfVariables);

        int T = 0;
        for (int timestep = 0; timestep < targets.length; timestep++) {
            Vec gradient_at_time = new DenseVector(totalNumOfVariables);
            for (int i = 0; i < outputNodes.size(); i++) {
                INode outputNode = outputNodes.get(i);
                ArrayList<Outcome> outcomesAtTime = networkHistory.getStateOfRecord(timestep, outputNode);
                HashMap<Outcome, Vec> gradientAtTime = networkGradient.get(timestep);
                HashMap<Outcome, Vec> probGradAtTime = probabilityGradient.get(timestep);
                Double target = targets[timestep][i];
                gradient_at_time
                        .mutableAdd(computeGradientOfOutput(outcomesAtTime, gradientAtTime, probGradAtTime, target));
            }
            if (gradient_at_time.nnz() != 0) {
                T++;
                gradient.mutableAdd(gradient_at_time);
            }
        }
        if (T == 0) {
            return gradient;
        } else {
            return gradient.divide(T);
        }

    }

    public void setTargets(Double[][] targets) {
        this.targets = targets;
    }

    public Double[][] getTargets() {
        return targets;
    }

    public double getTotalError(NetworkHistory networkHistory) {
        double error = 0;

        for (int i = 0; i < outputNodes.size(); i++) {
            for (int timestep = 0; timestep < targets.length; timestep++) {
                INode outputNode = outputNodes.get(i);
                ArrayList<Outcome> outcomesAtTime = networkHistory.getStateOfRecord(timestep, outputNode);
                Double target = targets[timestep][i];
                error += computeErrorOfOutput(outcomesAtTime, target);
            }
        }
        return error / targets.length;

    }

    protected Vec computeGradientOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime,
            HashMap<Outcome, Vec> probGradAtTime, Double target) {
        Vec gradient = new DenseVector(totalNumOfVariables);

        if (outcomesAtTime == null | target == null) {
            return gradient;
        }

        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);
        double probVolDerivative = probGradAtTime.values().stream().mapToDouble(vec -> vec.get(0)).sum();

        if (probabilityVolume*probabilityVolume == 0) {
            return gradient;
        }

        for (Outcome outcome : outcomesAtTime) {
            double error = errorFunction.error(outcome.activatedValue, target);
            double error_derivative = errorFunction.error_derivative(outcome.activatedValue, target);

            // accumulate jacobians
            Vec networkDerivative = gradientAtTime.get(outcome);
            double probabilityDerivative = probGradAtTime.get(outcome).get(0);

            double coef1 = probabilityDerivative * error * probabilityVolume;
            double coef2 = outcome.probability * error_derivative * probabilityVolume;
            double coef3 = outcome.probability * error * probVolDerivative;

            double sum = coef1 + coef2 - coef3;
            gradient.mutableAdd(networkDerivative.multiply(sum));
        }

        gradient.mutableDivide(probabilityVolume * probabilityVolume);
        return gradient;
    }

    protected double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target) {
        double error = 0;

        if (outcomesAtTime == null | target == null) {
            return error;
        }

        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);

        if (probabilityVolume == 0) {
            return error;
        }

        for (Outcome outcome : outcomesAtTime) {
            error += outcome.probability * errorFunction.error(outcome.activatedValue, target);
        }

        return error / probabilityVolume;
    }

}
