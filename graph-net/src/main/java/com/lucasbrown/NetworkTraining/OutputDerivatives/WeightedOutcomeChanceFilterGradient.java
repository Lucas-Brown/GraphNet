package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.INetworkGradient;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class WeightedOutcomeChanceFilterGradient extends GradientBase {

    private static double temperature = 1;
    private ErrorFunction errorFunction;

    public WeightedOutcomeChanceFilterGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables){
        super(network, networkGradientEvaluater, targets, totalNumOfVariables);
        this.errorFunction = errorFunction;
    }

    protected Vec computeGradientOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime, Double target) {
        Vec gradient = new DenseVector(totalNumOfVariables);
        
        if(outcomesAtTime == null || outcomesAtTime.isEmpty()){
            return gradient;
        }
        
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

            networkDerivative.multiply(target == null ? -1 : 1);
            assert networkDerivative.countNaNs() == 0;

            // accumulate jacobians
            gradient.mutableAdd(networkDerivative.multiply(errorOfOutput));
        }
        if(totalOutputError == 0){
            gradient.mutableMultiply(0);
            return gradient;
        }

        assert gradient.countNaNs() == 0;
        return gradient.divide(-totalOutputError);
    }

    protected double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target) {
        if(outcomesAtTime == null || outcomesAtTime.isEmpty()){
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

    
    // public static double[][] targetsToProbabilies(Double[][] targets){
    //     double[][] probs = new double[targets.length][];
    //     for (int i = 0; i < probs.length; i++) {
    //         probs[i] = new double[targets[i].length];
    //         for (int j = 0; j < probs[i].length; j++) {
    //             probs[i][j] = targets[i][j] == null ? 0 : 1;
    //         }
    //     }
    //     return probs;
    // }

}
