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
import jsat.linear.Matrix;
import jsat.linear.SingularValueDecomposition;
import jsat.linear.Vec;

public class NewtonTrainer extends Trainer {

    public double epsilon = 0.01;

    private int totalNumOfVariables;
    private HashMap<ITrainable, Integer> vectorNodeOffset;

    private Vec parameterDeltas;
    private Vec errorDerivative;
    private Matrix errorHessian;

    // consider a mask instead?
    private boolean[] isColumnEmpty;

    public NewtonTrainer(GraphNetwork network, ErrorFunction errorFunction) {
        super(network, errorFunction);

        InitializeOffsetMap();
    }

    private void InitializeOffsetMap() {
        vectorNodeOffset = new HashMap<>(allNodes.size());

        totalNumOfVariables = 0;
        for (ITrainable node : allNodes) {
            vectorNodeOffset.put(node, totalNumOfVariables);
            totalNumOfVariables += node.getNumberOfVariables();
        }
    }

    private int getLinearIndexOfWeight(ITrainable node, int key, int weight_index) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfWeight(key, weight_index);
    }

    private int getLinearIndexOfBias(ITrainable node, int key) {
        return vectorNodeOffset.get(node) + node.getLinearIndexOfBias(key);
    }

    @Override
    protected void computeErrorOfNetwork(boolean print_forward) {
        computeFullErrorDerivatives();
        computeDelta(print_forward);
    }

    private void computeFullErrorDerivatives() {
        for (int time = 0; time < inputs.length; time++) {
            for (ITrainable node : allNodes) {
                computeFullErrorDerivatives(node, time);
            }
        }
    }

    protected void computeFullErrorDerivatives(ITrainable node, int timestep) {
        ArrayList<Outcome> outcomes = networkHistory.getStateOfNode(timestep, node);

        if (outcomes == null || outcomes.isEmpty()) {
            return;
        }

        // initialize matrices
        for (Outcome outcome : outcomes) {
            outcome.errorJacobian = new DenseVector(totalNumOfVariables);
            outcome.errorHessian = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);
        }

        // Compute the Jacobians and Hessians
        for (Outcome outcome : outcomes) {
            Vec z_jacobi = computeJacobian(node, outcome);
            computeHessian(node, outcome, z_jacobi);
        }
    }

    protected Vec computeJacobian(ITrainable node, Outcome outcome) {
        // the Jacobian and Hessian of the input matrix will always be zero
        if (node instanceof InputNode) {
            return null;
        }

        Vec z_jacobi = new DenseVector(totalNumOfVariables);
        int key = outcome.binary_string;
        double[] weights = node.getWeights(key);

        // construct the jacobian for the net value (z)
        // starting with the direct derivative of z
        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            int idx = getLinearIndexOfWeight(node, key, i);
            z_jacobi.set(idx, outcome.sourceOutcomes[i].activatedValue);
        }
        int bias_idx = getLinearIndexOfBias(node, key);
        z_jacobi.set(bias_idx, 1);

        // incorporate previous jacobians
        for (int i = 0; i < weights.length; i++) {
            Outcome so = outcome.sourceOutcomes[i];
            Vec weighed_jacobi = so.errorJacobian.multiply(weights[i]);
            z_jacobi.mutableAdd(weighed_jacobi);
        }

        // apply to activated jacobi
        ActivationFunction activator = node.getActivationFunction();
        double activation_derivative = activator.derivative(outcome.netValue);
        outcome.errorJacobian = z_jacobi.multiply(activation_derivative);

        return z_jacobi;

    }

    /**
     * 
     * @param node
     * @param outcome
     * @param probabilityVolume
     */
    protected void computeHessian(ITrainable node, Outcome outcome, Vec z_jacobi) {
        int key = outcome.binary_string;
        double[] weights = node.getWeights(key);

        // use the net value jacobian to compute the activation jacobian
        ActivationFunction activator = node.getActivationFunction();
        double activation_derivative = activator.derivative(outcome.netValue);
        double activation_second_derivative = activator.secondDerivative(outcome.netValue);

        // construct hessian
        Matrix JJT = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);
        Matrix.OuterProductUpdate(JJT, z_jacobi, z_jacobi, 1);
        Matrix jacobi_chain = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);

        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            int idx = getLinearIndexOfWeight(node, key, i);
            Outcome so = outcome.sourceOutcomes[i];
            Vec jac = so.errorJacobian;
            jacobi_chain.getColumnView(idx).mutableAdd(jac);
        }

        jacobi_chain.mutableAdd(jacobi_chain.transpose());

        for (int i = 0; i < outcome.sourceOutcomes.length; i++) {
            Outcome so = outcome.sourceOutcomes[i];
            jacobi_chain.mutableAdd(so.errorHessian.multiply(weights[i]));
        }

        // finalize Hessian
        outcome.errorHessian = JJT.multiply(activation_second_derivative);
        outcome.errorHessian.mutableAdd(jacobi_chain.multiply(activation_derivative));
    }

    protected void computeDelta(boolean print_forward) {
        errorDerivative = new DenseVector(totalNumOfVariables);
        errorHessian = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);

        computeErrorOfOutput(print_forward);

        // double check hesssian symmetry
        for (int i = 0; i < totalNumOfVariables; i++) {
            for (int j = i + 1; j < totalNumOfVariables; j++) {
                assert errorHessian.get(i, j) == errorHessian.get(j, i);
            }
        }

        // computeDeltaSeparateZeros();
        computeDeltaWithZeros();

    }

    private void computeDeltaWithZeros() {
        Matrix eta_Matrix = DenseMatrix.eye(totalNumOfVariables).multiply(0);
        SingularValueDecomposition decomposition = new SingularValueDecomposition(errorHessian.add(eta_Matrix));

        parameterDeltas = decomposition.solve(errorDerivative);
        parameterDeltas.mutableMultiply(epsilon);

        // ensure that we aim for the minimum instead of the maximum
        // for (int i = 0; i < parameterDeltas.rows(); i++) {
        // parameterDeltas.set(i, 0,
        // Math.abs(parameterDeltas.get(i, 0)) * Math.signum(errorDerivative.get(i,
        // 0)));
        // }
    }

    private void computeDeltaSeparateZeros() {
        Vec errorDerivativeNonZero = removeZeroColumns();

        SingularValueDecomposition decomposition = new SingularValueDecomposition(errorHessian);

        // if the matrix can't be practically inverted, use a simple gradient step
        // if(Math.abs(decomposition.det()) < 1E-6){
        // parameterDeltas = errorDerivative.multiply(epsilon/inputs.length);
        // return;
        // }

        Vec non_zero_deltas = decomposition.solve(errorDerivativeNonZero);
        non_zero_deltas.mutableMultiply(epsilon);

        // ensure that we aim for the minimum instead of the maximum
        for (int i = 0; i < non_zero_deltas.length(); i++) {
            non_zero_deltas.set(i,
                    Math.abs(non_zero_deltas.get(i)) * Math.signum(errorDerivativeNonZero.get(i)));
        }

        // recombine deltas
        parameterDeltas = new DenseVector(totalNumOfVariables);
        int non_empty_idx = 0;
        for (int i = 0; i < totalNumOfVariables; i++) {
            double value;
            if (isColumnEmpty[i]) {
                value = errorDerivative.get(i) * epsilon * 0.001;
            } else {
                value = non_zero_deltas.get(non_empty_idx++);
            }

            parameterDeltas.set(i, value);
        }
    }

    private Vec removeZeroColumns() {

        isColumnEmpty = new boolean[totalNumOfVariables];
        int n_nnz = 0;
        for (int i = 0; i < totalNumOfVariables; i++) {
            isColumnEmpty[i] = errorHessian.getColumn(i).nnz() == 0;
            if (!isColumnEmpty[i]) {
                n_nnz++;
            }
        }

        Vec new_derivative = new DenseVector(n_nnz);
        Matrix new_hessian = new DenseMatrix(n_nnz, n_nnz);

        int non_zero_index = 0;
        for (int i = 0; i < totalNumOfVariables; i++) {
            if (!isColumnEmpty[i]) {
                new_derivative.set(non_zero_index, errorDerivative.get(i));

                Vec new_vec = new_hessian.getColumnView(non_zero_index);
                Vec old_vec = errorHessian.getColumn(i);
                int non_zero_index_2 = 0;
                for (int j = 0; j < totalNumOfVariables; j++) {
                    if (!isColumnEmpty[j]) {
                        new_vec.set(non_zero_index_2++, old_vec.get(j));
                    }
                }
                non_zero_index++;
            }
        }

        errorHessian = new_hessian;
        return new_derivative;
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

        assert Double.isFinite(total_error.getAverage());
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

        probabilityVolume = 1;

        for (Outcome outcome : outcomes) {
            // if (timestep == this.timestep) {
            // outcome.passRate.add(1 - 1d / timestep, 1);
            // } else {
            outcome.passRate.add(1, 1);
            // }

            double error_derivative = errorFunction.error_derivative(outcome.activatedValue, target);
            double error_second_derivative = errorFunction.error_second_derivative(outcome.activatedValue, target);
            double prob = outcome.probability / probabilityVolume;

            // accumulate jacobians
            errorDerivative.mutableAdd(outcome.errorJacobian.multiply(prob * error_derivative));

            // accumulate hessian
            Matrix JJT = new DenseMatrix(totalNumOfVariables, totalNumOfVariables);
            Matrix.OuterProductUpdate(JJT, outcome.errorJacobian, outcome.errorJacobian, 1);
            errorHessian.mutableAdd(JJT.multiply(prob * error_second_derivative));
            errorHessian.mutableAdd(outcome.errorHessian.multiply(prob * error_derivative));

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
