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

public class OutcomeChanceFilterGradient implements IGradient{

    private Double[][] targets;
    private ErrorFunction errorFunction;
    private INetworkGradient networkGradientEvaluater;
    protected ArrayList<OutputNode> outputNodes;
    private Vec gradient;
    private int totalNumOfVariables;

    public OutcomeChanceFilterGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables){
        this.targets = targets;
        this.errorFunction = errorFunction;
        this.networkGradientEvaluater = networkGradientEvaluater;
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
                for (double d : gradient.arrayCopy()) {
                    assert Double.isFinite(d);
                }
                
            }
        }
        return gradient;
    }
    
    protected void computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime, Double target) {
        if(outcomesAtTime == null){
            return;
        }
        
        double probability = target == null ? 0 : 1; 

        for (Outcome outcome : outcomesAtTime) {

            double error = errorFunction.error_derivative(outcome.probability, probability);

            // accumulate jacobians
            Vec networkDerivative = gradientAtTime.get(outcome);
            gradient.mutableAdd(networkDerivative.multiply(error));
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

    public static double[][] targetsToProbabilies(Double[][] targets){
        double[][] probs = new double[targets.length][];
        for (int i = 0; i < probs.length; i++) {
            probs[i] = new double[targets[i].length];
            for (int j = 0; j < probs[i].length; j++) {
                probs[i][j] = targets[i][j] == null ? 0 : 1;
            }
        }
        return probs;
    }
}
