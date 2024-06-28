package com.lucasbrown.GraphNetwork.Global.Trainers;

import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.NetworkTraining.History;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class ADAMTrainer implements ISolver{

    protected IGradient gradientEvaluator;
    private int totalNumOfVariables;

    public double alpha = 0.001;
    public double epsilon = 1E-8;
    public double beta_1 = 0.99;
    public double beta_2 = 0.999;

    private int t;

    private Vec parameterDeltas;

    private Vec m; // biased first-moment estimate
    private Vec v; // biased second-moment estimate
    private Vec m_hat; // bias corrected first-moment
    private Vec v_hat; // bias corrected second-moment

    public ADAMTrainer(IGradient gradientEvaluator, int totalNumOfVariables) {
        this.gradientEvaluator = gradientEvaluator;
        this.totalNumOfVariables = totalNumOfVariables;

        t = 0;
        m = new DenseVector(totalNumOfVariables);
        v = new DenseVector(totalNumOfVariables);
        m_hat = new DenseVector(totalNumOfVariables);
        v_hat = new DenseVector(totalNumOfVariables);
    }


    @Override
    public Vec solve(History<Outcome, INode> history) {
        Vec errorDerivative = gradientEvaluator.computeGradient(history);
        ADAM_step(errorDerivative);
        return parameterDeltas;
    }

    private void ADAM_step(Vec errorDerivative) {
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



}
