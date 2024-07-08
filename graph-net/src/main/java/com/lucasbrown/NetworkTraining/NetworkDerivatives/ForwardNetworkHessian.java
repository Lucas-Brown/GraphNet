package com.lucasbrown.NetworkTraining.NetworkDerivatives;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.Map.Entry;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ValueCombinators.IValueCombinator;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;
import com.lucasbrown.NetworkTraining.Trainers.WeightsLinearizer;

import jsat.linear.DenseMatrix;
import jsat.linear.Matrix;
import jsat.linear.Vec;

/**
 * Computes the hessian using a forward pass
 */
public class ForwardNetworkHessian extends ForwardNetworkGradient implements INetworkHessian {

    private ArrayList<HashMap<Outcome, Matrix>> hessianThroughTime;

    public ForwardNetworkHessian(WeightsLinearizer linearizer) {
        super(linearizer);
    }

    @Override
    public ArrayList<HashMap<Outcome, Matrix>> getHessian(NetworkHistory networkHistory) {
        this.networkHistory = networkHistory;
        int n_steps = networkHistory.getNumberOfTimesteps();
        hessianThroughTime = new ArrayList<>(n_steps);

        for (int timestep = 0; timestep < n_steps; timestep++) {
            hessianThroughTime.add(getHessianAtTime(timestep));
        }

        return hessianThroughTime;
    }

    private HashMap<Outcome, Matrix> getHessianAtTime(int timestep) {
        HashMap<Outcome, Matrix> hessianMap = new HashMap<>();
        HashMap<INode, ArrayList<Outcome>> outcomeMap = networkHistory.getStateAtTimestep(timestep);
        for (Entry<INode, ArrayList<Outcome>> entry : outcomeMap.entrySet()) {
            INode node = entry.getKey();
            for (Outcome outcome : entry.getValue()) {
                hessianMap.put(outcome, getHessianOfOutcome(node, outcome));
            }
        }
        return hessianMap;
    }

    /**
     * 
     * @param node
     * @param outcome
     * @param probabilityVolume
     */
    protected Matrix getHessianOfOutcome(INode node, Outcome outcome) {
        castTrainingDataToHess(outcome);
        int totalNumOfVariables = linearizer.totalNumOfVariables;
        if (node instanceof InputNode) {
            Matrix matrix = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);
            ((JacobiAndHess) outcome.trainingData).hessian = matrix;
            return matrix;
        }

        Vec z_jacobi = getZJacobi(node, outcome);

        int key = outcome.binary_string;
        IValueCombinator combinator = node.getValueCombinator();
        double[] weights = combinator.getWeights(key);

        // use the net value jacobian to compute the activation jacobian
        ActivationFunction activator = node.getActivationFunction();
        double activation_derivative = activator.derivative(outcome.netValue);
        double activation_second_derivative = activator.secondDerivative(outcome.netValue);

        // construct hessian
        Matrix JJT = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);
        Matrix.OuterProductUpdate(JJT, z_jacobi, z_jacobi, 1);
        Matrix jacobi_chain = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);

        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            int idx = linearizer.getLinearIndexOfWeight(node, key, i);
            Outcome so = outcome.sourceOutcomes[i];
            Vec jac = ((JacobiAndHess) so.trainingData).jacobian;
            jacobi_chain.getColumnView(idx).mutableAdd(jac);
        }

        jacobi_chain.mutableAdd(jacobi_chain.transpose());

        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            Outcome so = outcome.sourceOutcomes[i];
            jacobi_chain.mutableAdd(((JacobiAndHess) so.trainingData).hessian.multiply(weights[i]));
        }

        // finalize Hessian
        Matrix hessian = JJT.multiply(activation_second_derivative);
        hessian.mutableAdd(jacobi_chain.multiply(activation_derivative));
        ((JacobiAndHess) outcome.trainingData).hessian = hessian;
        return hessian;
    }

    private void castTrainingDataToHess(Outcome outcome) {
        Vec jacobi = (Vec) outcome.trainingData;
        JacobiAndHess jah = new JacobiAndHess();
        jah.jacobian = jacobi;
        outcome.trainingData = jah;
    }

    private Vec getZJacobi(INode node, Outcome outcome) {
        double derivative = node.getActivationFunction().derivative(outcome.netValue);
        return ((JacobiAndHess) outcome.trainingData).jacobian.divide(derivative);
    }

    private static class JacobiAndHess {
        Vec jacobian;
        Matrix hessian;
    }
}
