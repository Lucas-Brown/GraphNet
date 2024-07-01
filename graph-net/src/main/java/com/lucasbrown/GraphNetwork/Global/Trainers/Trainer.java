package com.lucasbrown.GraphNetwork.Global.Trainers;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.WeightedAverage;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;

import jsat.linear.Vec;

public class Trainer implements ITrainer{

    private WeightsLinearizer weightLinearizer;
    private FilterLinearizer filterLinearizer;
    private NetworkInputEvaluater networkEvaluater;
    private ISolver weightsSolver;
    private ISolver probabilitySolver;

    private Vec weightsDeltas;
    private Vec probabilityDeltas;

    protected Double[][] inputs;
    protected Double[][] targets;

    protected WeightedAverage total_error;

    public Trainer(NetworkInputEvaluater networkEvaluater, ISolver weightsSolver, ISolver probabilitySolver, WeightsLinearizer weightLinearizer, FilterLinearizer filterLinearizer) {
        this.weightsSolver = weightsSolver;
        this.probabilitySolver = probabilitySolver;

        this.weightLinearizer = weightLinearizer;
        this.filterLinearizer = filterLinearizer;
        this.networkEvaluater = networkEvaluater;

        total_error = new WeightedAverage();
    }

    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][] inputs, Double[][] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public void trainNetwork(int steps, int print_interval) {
        while (steps-- > 0) {
            trainingStep(steps % print_interval == 0);
        }
    }

    public void trainingStep(boolean print_forward) {
        History<Outcome, INode> history = networkEvaluater.computeNetworkInference();
        weightsDeltas = weightsSolver.solve(history);
        probabilityDeltas = probabilitySolver.solve(history);

        applyWeightDeltas();
        applyProbabilityDeltas();
    }

    private void applyWeightDeltas() {
        weightLinearizer.allNodes.forEach(this::applyErrorSignalsToNode);
    }

    private void applyErrorSignalsToNode(ITrainable node) {
        double[] allDeltas = weightsDeltas.arrayCopy();
        double[] gradient = weightLinearizer.nodeSlice(node, allDeltas);
        node.applyDelta(gradient);
    }

    private void applyProbabilityDeltas() {
        filterLinearizer.allFilters.forEach(this::applyParameterUpdate);
    }

    private void applyParameterUpdate(IFilter filter) {
        double[] gradient = filterLinearizer.filterSlice(filter, probabilityDeltas.arrayCopy());
        filter.applyAdjustableParameterUpdate(gradient);
    }

    public static Trainer getDefaultTrainer(GraphNetwork network, Double[][] inputs, Double[][] targets){
        WeightsLinearizer weightLinearizer = new WeightsLinearizer(network);
        FilterLinearizer filterLinearizer = new FilterLinearizer(network);
        NetworkInputEvaluater networkEvaluater = new NetworkInputEvaluater(network, inputs);

        ErrorFunction erf = new ErrorFunction.MeanSquaredError();

        DirectNetworkGradient netGradient = new DirectNetworkGradient(network, new ForwardNetworkGradient(weightLinearizer), targets, erf, weightLinearizer.totalNumOfVariables, true); 
        ADAMTrainer weightsTrainer = new ADAMTrainer(netGradient, weightLinearizer.totalNumOfVariables); 

        DirectFilterGradient filterGradient = new DirectFilterGradient(network, new ForwardFilterGradient(filterLinearizer), targets, erf, filterLinearizer.totalNumOfVariables);
        ADAMTrainer filterTrainer = new ADAMTrainer(filterGradient, filterLinearizer.totalNumOfVariables);

        return new Trainer(networkEvaluater, weightsTrainer, filterTrainer, weightLinearizer, filterLinearizer);
    }
}
