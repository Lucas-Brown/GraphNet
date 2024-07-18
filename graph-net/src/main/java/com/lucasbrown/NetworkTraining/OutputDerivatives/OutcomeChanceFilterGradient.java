package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.INetworkGradient;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class OutcomeChanceFilterGradient extends GradientBase {

    private ErrorFunction errorFunction;

    public OutcomeChanceFilterGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater,
            Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables) {
        super(network, networkGradientEvaluater, targets, totalNumOfVariables);
        this.errorFunction = errorFunction;
    }

    protected Vec computeGradientOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime,
            Double target) {
        Vec gradient = new DenseVector(totalNumOfVariables);

        if (outcomesAtTime == null || outcomesAtTime.isEmpty()) {
            return gradient;
        }

        double probabilityVolume = IGradient.getProbabilityVolume(outcomesAtTime);

        boolean is_min = target == null;

        // if (is_min && probabilityVolume == 1 || !is_min && probabilityVolume == 0) {
        //     return gradient;
        // }

        for (Outcome outcome : outcomesAtTime) {
            // accumulate jacobians
            Vec jac = gradientAtTime.get(outcome);
            // if(outcome.probability == 0){
            //     assert jac.nnz() == 0; 
            //     continue;
            // }
            gradient.mutableAdd(jac);//.divide(outcome.probability));
        }

        return is_min ? gradient : gradient.multiply(-1);

        // if (is_min) {
        //     return gradient.multiply(probabilityVolume);
        // } else {
        //     return gradient.multiply(1-probabilityVolume);
        // }
    }

    protected double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target) {
        if (outcomesAtTime == null || outcomesAtTime.isEmpty()) {
            return 0;
        }

        double error = IGradient.getProbabilityVolume(outcomesAtTime);

        return target == null ? error : 1 - error;
    }

    // public static double[][] targetsToProbabilies(Double[][] targets){
    // double[][] probs = new double[targets.length][];
    // for (int i = 0; i < probs.length; i++) {
    // probs[i] = new double[targets[i].length];
    // for (int j = 0; j < probs[i].length; j++) {
    // probs[i][j] = targets[i][j] == null ? 0 : 1;
    // }
    // }
    // return probs;
    // }

}
