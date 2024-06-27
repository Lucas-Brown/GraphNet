package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;

import jsat.linear.DenseMatrix;
import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class ADAMTrainer extends Trainer {

    public double alpha = 0.001;
    public double epsilon = 1E-8;
    public double beta_1 = 0.99;
    public double beta_2 = 0.999;

    private int t;

    private Vec parameterDeltas;
    private Vec errorDerivative;

    private Vec m; // biased first-moment estimate
    private Vec v; // biased second-moment estimate
    private Vec m_hat; // bias corrected first-moment
    private Vec v_hat; // bias corrected second-moment

    public ADAMTrainer(GraphNetwork network, ErrorFunction errorFunction) {
        super(network, errorFunction);

        t = 0;
        InitializeOffsetMap();

        m = new DenseVector(totalNumOfVariables);
        v = new DenseVector(totalNumOfVariables);
        m_hat = new DenseVector(totalNumOfVariables);
        v_hat = new DenseVector(totalNumOfVariables);
    }



    @Override
    protected void computeErrorOfNetwork(boolean print_forward) {
        computeFullErrorDerivatives();
        computeDelta(print_forward);
    }


    protected void computeDelta(boolean print_forward) {
        errorDerivative = new DenseVector(totalNumOfVariables);

        computeErrorOfOutput(print_forward);
        ADAM_step();
    }

    private void ADAM_step() {
        t++;
        m = m.multiply(beta_1).add(errorDerivative.multiply(1 - beta_1));
        v = v.multiply(beta_2).add(errorDerivative.pairwiseMultiply(errorDerivative).multiply(1 - beta_2));
        m_hat = m.divide(1 - Math.pow(beta_1, t));
        v_hat = v.divide(1 - Math.pow(beta_2, t));

        parameterDeltas = new DenseVector(totalNumOfVariables);
        for (int i = 0; i < totalNumOfVariables; i++) {
            double denom = Math.sqrt(v_hat.get(i)) + epsilon;
            parameterDeltas.set(i, alpha * m_hat.get(i) / denom);
        }
    }

    private double getProbabilityVolume(ArrayList<Outcome> outcomes) {
        return outcomes.stream().mapToDouble(outcome -> outcome.probability).sum();
    }

    protected void computeErrorOfOutput(boolean print_forward) {
        for (int time = inputs.length - 1; time > 0; time--) {
            for (int i = 0; i < outputNodes.size(); i++) {
                computeErrorOfOutput(outputNodes.get(i), time, targets[time][i]);
            }
        }

        //assert Double.isFinite(total_error.getAverage());
        if (print_forward) {
            System.out.println(total_error.getAverage());
        }
        total_error.reset();
    }

    protected void computeErrorOfOutput(OutputNode node, int timestep, Double target) {
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node);
        if (outcomes == null) {
            return;
        }

        if (target == null) {
            for (Outcome outcome : outcomes) {
                outcome.passRate.add(0, 1);
            }
            return;
        }

        double probabilityVolume = getProbabilityVolume(outcomes);
        if (probabilityVolume == 0) {
            return;
        }

        // probabilityVolume = 1;

        for (Outcome outcome : outcomes) {
            // if (timestep == this.timestep) {
            // outcome.passRate.add(1 - 1d / timestep, 1);
            // } else {
            outcome.passRate.add(1, 1);
            // }

            double error_derivative = errorFunction.error_derivative(outcome.activatedValue, target);
            double prob = outcome.probability / probabilityVolume;

            // accumulate jacobians
            errorDerivative.mutableAdd(outcome.errorJacobian.multiply(prob * error_derivative));

            double error = errorFunction.error(outcome.activatedValue, target);
            // assert Double.isFinite(errorFunction.error(outcome.activatedValue, target));
            total_error.add(error, outcome.probability);
        }

    }

    @Override
    protected void applyErrorSignals() {
        parameterDeltas.mutableMultiply(epsilon);
        allNodes.forEach(this::applyErrorSignalsToNode);
    }

    public void applyErrorSignalsToNode(ITrainable node) {
        double[] allDeltas = parameterDeltas.arrayCopy();
        double[] gradient = gradientOfNode(node, allDeltas);
        node.applyDelta(gradient);
    }

    private double[] gradientOfNode(ITrainable node, double[] allDeltas) {
        int startIdx = vectorNodeOffset.get(node);
        int length = node.getNumberOfVariables();
        double[] gradient = new double[length];
        System.arraycopy(allDeltas, startIdx, gradient, 0, length);
        return gradient;
    }
}
