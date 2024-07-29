package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.INetworkGradient;
import com.lucasbrown.NetworkTraining.OutputDerivatives.ErrorFunction.CrossEntropy;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class WeightedOutcomeChanceFilterGradient implements IGradient {

    protected Double[][] targets;
    protected INetworkGradient networkGradientEvaluater;
    protected Vec gradient;
    protected ArrayList<OutputNode> outputNodes;
    protected int totalNumOfVariables;

    
    private Vec probGrad, valueGrad;
    private double probError, valueError;

    private static double temperature = 1;
    private ErrorFunction errorFunction;
    private static final CrossEntropy crossEntropy = new CrossEntropy();

    public WeightedOutcomeChanceFilterGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables){
        this.targets = targets;
        this.networkGradientEvaluater = networkGradientEvaluater;
        this.totalNumOfVariables = totalNumOfVariables;
        this.errorFunction = errorFunction;

        outputNodes = network.getOutputNodes();
    }

    
    @Override
    public Vec computeGradient(NetworkHistory networkHistory) {
        ArrayList<HashMap<Outcome, Vec>> networkGradient = networkGradientEvaluater.getGradient(networkHistory);
        gradient = new DenseVector(totalNumOfVariables);

        gradientOfTargets(networkHistory, networkGradient);
        gradientOfValues(networkHistory, networkGradient);

        gradient.mutableAdd(probGrad);
        gradient.mutableAdd(valueGrad);

        return gradient;

    }

    private void gradientOfTargets(NetworkHistory networkHistory, ArrayList<HashMap<Outcome, Vec>> networkGradient){
        
        final int out_size = outputNodes.size();
        Vec[] gradient_prob = new Vec[out_size];
        int[] T_prob = new int[out_size];

        probGrad = new DenseVector(totalNumOfVariables);

        if(out_size == 0){
            return;
        }

        for (int i = 0; i < out_size; i++) {
            gradient_prob[i] = new DenseVector(totalNumOfVariables);
        }

        // loop over all output nodes at every timestep
        for (int timestep = 0; timestep < targets.length; timestep++) {

            for (int i = 0; i < out_size; i++) {
                INode outputNode = outputNodes.get(i);
                ArrayList<Outcome> outcomesAtTime = networkHistory.getStateOfRecord(timestep, outputNode);
                HashMap<Outcome, Vec> gradientAtTime = networkGradient.get(timestep);
                Double target = targets[timestep][i];
                boolean is_value = target != null;

                // skip when there's no outcome
                if(outcomesAtTime == null || outcomesAtTime.isEmpty()){
                    continue;
                }

                // probabilty component
                gradient_prob[i].mutableAdd(gradientOfTarget(outcomesAtTime, gradientAtTime, is_value ? 1 : 0));
                T_prob[i]++;
            }

        }

        for (int i = 0; i < out_size; i++) {
            if(T_prob[i] == 0) continue;

            probGrad.mutableAdd(gradient_prob[i].divide(T_prob[i]));
        }
    }

    private void gradientOfValues(NetworkHistory networkHistory, ArrayList<HashMap<Outcome, Vec>> networkGradient){

        final int out_size = outputNodes.size();
        Vec[] gradient_value = new Vec[out_size];
        int[] T_value = new int[out_size];

        valueGrad = new DenseVector(totalNumOfVariables);

        if(out_size == 0){
            return;
        }

        for (int i = 0; i < out_size; i++) {
            gradient_value[i] = new DenseVector(totalNumOfVariables);
        }

        // loop over all output nodes at every timestep
        for (int timestep = 0; timestep < targets.length; timestep++) {

            for (int i = 0; i < out_size; i++) {
                INode outputNode = outputNodes.get(i);
                ArrayList<Outcome> outcomesAtTime = networkHistory.getStateOfRecord(timestep, outputNode);
                HashMap<Outcome, Vec> gradientAtTime = networkGradient.get(timestep);
                Double target = targets[timestep][i];
                boolean is_value = target != null;

                // skip when there's no outcome
                if(outcomesAtTime == null || outcomesAtTime.isEmpty()){
                    continue;
                }

                // value component 
                if(!is_value){
                    continue;
                }
                
                Vec gradient_value_at_time = weightedValueGradient(outcomesAtTime, gradientAtTime, (double) target);
                gradient_value[i].mutableAdd(gradient_value_at_time);
                T_value[i]++;
            }

        }

        for (int i = 0; i < out_size; i++) {
            if(T_value[i] == 0) continue;

            valueGrad.mutableAdd(gradient_value[i].divide(T_value[i]));
        }
    }

    private Vec weightedValueGradient(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime, double target) {
        Vec gradient = new DenseVector(totalNumOfVariables);

        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);

        if(probabilityVolume == 0){
            return gradient;
        }

        double totalOutputError = 0;
        for (Outcome outcome : outcomesAtTime) {

            double errorOfOutput = errorFunction.error(outcome.activatedValue, target);
            errorOfOutput = Math.exp(-errorOfOutput/temperature);
            totalOutputError += errorOfOutput;

            Vec networkDerivative = gradientAtTime.get(outcome);
            if(networkDerivative.nnz() == 0){
                continue;
            }

            assert networkDerivative.countNaNs() == 0;

            // accumulate jacobians
            gradient.mutableAdd(networkDerivative.multiply(errorOfOutput));
        }
        if(totalOutputError == 0){
            gradient.mutableMultiply(0);
            return gradient;
        }

        assert gradient.countNaNs() == 0;
        gradient.mutableDivide(-totalOutputError*probabilityVolume);
        return gradient.multiply(1);
    }

    private Vec gradientOfTarget(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime, double targetProbability) {
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


    @Override
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

    protected double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target) {
        if(outcomesAtTime == null || outcomesAtTime.isEmpty() || target == null){
            return 0;
        }
        
        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);

        if(probabilityVolume == 0){
            return 0;
        }

        double error = 0;
        double totalOutputError = 0;
        for (Outcome outcome : outcomesAtTime) {
            double errorOfOutput = errorFunction.error(outcome.activatedValue, target);
            errorOfOutput = Math.exp(-errorOfOutput/temperature);
            totalOutputError += errorOfOutput;
            if(target == null){
                error += errorOfOutput * (1 - outcome.probability);
            }
            else{
                error += errorOfOutput * (outcome.probability);
            }
            
        }
        
        if(totalOutputError == 0){
            return 0;
        }
        return -error/totalOutputError;
    }

    private void localIterateOverHistory(NetworkHistory networkHistory, HistoryIteratorFunction histFunc){
        IGradient.iterateOverHistory(targets, outputNodes, networkHistory, histFunc);
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
