package com.lucasbrown.NetworkTraining.OutputDerivatives;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.NetworkDerivatives.INetworkGradient;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class ErrorFilterGradient extends GradientBase {

    private ErrorFunction errorFunction;

    public ErrorFilterGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets, ErrorFunction errorFunction, int totalNumOfVariables){
        super(network, networkGradientEvaluater, targets, totalNumOfVariables);
        this.errorFunction = errorFunction;
    }

    protected Vec computeGradientOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime, Double target) {
        Vec gradient = new DenseVector(totalNumOfVariables); 
        
        if(outcomesAtTime == null || target == null){
            return gradient;
        }

        for (Outcome outcome : outcomesAtTime) {

            double error = -errorFunction.error(outcome.activatedValue, target);

            // accumulate jacobians
            Vec networkDerivative = gradientAtTime.get(outcome);
            gradient.mutableAdd(networkDerivative.multiply(error));
        }

        return gradient;
    }

    @Override
    protected double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target) {
        double error = 0;
        
        if(outcomesAtTime == null || target == null){
            return error;
        }

        for (Outcome outcome : outcomesAtTime) {

            error += errorFunction.error(outcome.activatedValue, target);
        }

        return error;
    }

}
