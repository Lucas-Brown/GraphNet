package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class OutcomeChanceFilterGradient extends GradientBase {

    private ErrorFunction errorFunction;

    public OutcomeChanceFilterGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables){
        super(network, networkGradientEvaluater, targets, totalNumOfVariables);
        this.errorFunction = errorFunction;
    }

    protected Vec computeGradientOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime, Double target) {
        Vec gradient = new DenseVector(totalNumOfVariables);
        
        if(outcomesAtTime == null || outcomesAtTime.isEmpty()){
            return gradient;
        }
        
        double probability = target == null ? 0 : 1; 


        for (Outcome outcome : outcomesAtTime) {

            double error_derivative = errorFunction.error_derivative(outcome.probability, probability);

            // accumulate jacobians
            Vec networkDerivative = gradientAtTime.get(outcome);
            gradient.mutableAdd(networkDerivative.multiply(error_derivative));
        }

        return gradient.divide(outcomesAtTime.size());
    }

    protected double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target) {
        if(outcomesAtTime == null || outcomesAtTime.isEmpty()){
            return 0;
        }

        double probability = target == null ? 0 : 1; 


        double error = 0;
        for (Outcome outcome : outcomesAtTime) {
            error += errorFunction.error(outcome.probability, probability);
        }
        return error/outcomesAtTime.size();
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
