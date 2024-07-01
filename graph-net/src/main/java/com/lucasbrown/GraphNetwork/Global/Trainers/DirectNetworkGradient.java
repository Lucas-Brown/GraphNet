package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class DirectNetworkGradient implements IGradient{

    private Double[][] targets;
    private ErrorFunction errorFunction;
    private INetworkGradient networkGradientEvaluater;
    protected ArrayList<OutputNode> outputNodes;
    private boolean normalize;
    private Vec gradient;
    private int totalNumOfVariables;

    public DirectNetworkGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables, boolean normalize){
        this.targets = targets;
        this.errorFunction = errorFunction;
        this.networkGradientEvaluater = networkGradientEvaluater;
        this.normalize = normalize;
        this.totalNumOfVariables = totalNumOfVariables;
        outputNodes = network.getOutputNodes();
    }

    @Override
    public Vec computeGradient(History<Outcome, INode> networkHistory) {
        ArrayList<HashMap<Outcome, Vec>> networkGradient = networkGradientEvaluater.getGradient(networkHistory); 
        gradient = new DenseVector(totalNumOfVariables);

        for(int i = 0; i < outputNodes.size(); i++){
            for(int timestep = 0; timestep < targets.length; timestep++)
            {
                INode outputNode = outputNodes.get(i);
                ArrayList<Outcome> outcomesAtTime = networkHistory.getStateOfRecord(timestep, outputNode);
                HashMap<Outcome, Vec> gradientAtTime = networkGradient.get(timestep);
                Double target = targets[timestep][i];
                computeErrorOfOutput(outcomesAtTime, gradientAtTime, target);
            }
        }
        return gradient;
    }

    
    private double getProbabilityVolume(ArrayList<Outcome> outcomes) {
        return outcomes.stream().mapToDouble(outcome -> outcome.probability).sum();
    }

    
    protected void computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime, Double target) {
        if (outcomesAtTime == null) {
            return;
        }

        double probabilityVolume = getProbabilityVolume(outcomesAtTime);
        
        if (probabilityVolume == 0) {
            return;
        }

        if(!normalize){
            probabilityVolume = 1;
        }

        for (Outcome outcome : outcomesAtTime) {
            double error_derivative = errorFunction.error_derivative(outcome.activatedValue, target);
            double prob = outcome.probability / probabilityVolume;

            // accumulate jacobians
            Vec networkDerivative = gradientAtTime.get(outcome);
            gradient.mutableAdd(networkDerivative.multiply(prob * error_derivative));

        }

    }

    
}
