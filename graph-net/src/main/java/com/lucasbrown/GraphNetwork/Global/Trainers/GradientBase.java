package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public abstract class GradientBase implements IGradient {

    private Double[][] targets;
    private INetworkGradient networkGradientEvaluater;
    private Vec gradient;
    protected ArrayList<OutputNode> outputNodes;
    protected int totalNumOfVariables;

    public GradientBase(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets,
            int totalNumOfVariables) {
        this.targets = targets;
        this.networkGradientEvaluater = networkGradientEvaluater;
        this.totalNumOfVariables = totalNumOfVariables;
        outputNodes = network.getOutputNodes();
    }

    @Override
    public Vec computeGradient(History<Outcome, INode> networkHistory) {
        ArrayList<HashMap<Outcome, Vec>> networkGradient = networkGradientEvaluater.getGradient(networkHistory);
        gradient = new DenseVector(totalNumOfVariables);

        int T = 0;
        for (int timestep = 0; timestep < targets.length; timestep++) {
            Vec gradient_at_time = new DenseVector(totalNumOfVariables);
            for (int i = 0; i < outputNodes.size(); i++) {
                INode outputNode = outputNodes.get(i);
                ArrayList<Outcome> outcomesAtTime = networkHistory.getStateOfRecord(timestep, outputNode);
                HashMap<Outcome, Vec> gradientAtTime = networkGradient.get(timestep);
                Double target = targets[timestep][i];
                gradient_at_time.mutableAdd(computeGradientOfOutput(outcomesAtTime, gradientAtTime, target));
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

    @Override
    public void setTargets(Double[][] targets) {
        this.targets = targets;
    }

    @Override
    public Double[][] getTargets() {
        return targets;
    }

    @Override
    public double getTotalError(History<Outcome, INode> networkHistory) {
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

    public static double getProbabilityVolume(ArrayList<Outcome> outcomes) {
        return outcomes.stream().mapToDouble(outcome -> outcome.probability).sum();
    }

    protected abstract Vec computeGradientOfOutput(ArrayList<Outcome> outcomesAtTime,
            HashMap<Outcome, Vec> gradientAtTime, Double target);

    protected abstract double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target);
}
