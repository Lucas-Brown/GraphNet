package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class DirectNetworkGradient extends GradientBase {

    private ErrorFunction errorFunction;
    private boolean normalize;

    public DirectNetworkGradient(GraphNetwork network, INetworkGradient networkGradientEvaluater, Double[][] targets,
            ErrorFunction errorFunction, int totalNumOfVariables, boolean normalize) {
        super(network, networkGradientEvaluater, targets, totalNumOfVariables);
        this.errorFunction = errorFunction;
        this.normalize = normalize;
    }


    @Override
    protected Vec computeGradientOfOutput(ArrayList<Outcome> outcomesAtTime, HashMap<Outcome, Vec> gradientAtTime,
            Double target) {
        Vec gradient = new DenseVector(totalNumOfVariables);

        if (outcomesAtTime == null | target == null) {
            return gradient;
        }

        double probabilityVolume = GradientBase.getProbabilityVolume(outcomesAtTime);

        if (probabilityVolume == 0) {
            return gradient;
        }

        if (!normalize) {
            probabilityVolume = 1;
        }

        for (Outcome outcome : outcomesAtTime) {
            double error_derivative = errorFunction.error_derivative(outcome.activatedValue, target);
            double prob = outcome.probability / probabilityVolume;

            // accumulate jacobians
            Vec networkDerivative = gradientAtTime.get(outcome);
            gradient.mutableAdd(networkDerivative.multiply(prob * error_derivative));

        }

        return gradient;
    }

    @Override
    protected double computeErrorOfOutput(ArrayList<Outcome> outcomesAtTime, Double target) {
        double error = 0;

        if (outcomesAtTime == null | target == null) {
            return error;
        }

        double probabilityVolume = GradientBase.getProbabilityVolume(outcomesAtTime);

        if (probabilityVolume == 0) {
            return error;
        }

        if (!normalize) {
            probabilityVolume = 1;
        }

        for (Outcome outcome : outcomesAtTime) {
            error += outcome.probability * errorFunction.error(outcome.activatedValue, target);
        }

        return error / probabilityVolume;
    }

}
