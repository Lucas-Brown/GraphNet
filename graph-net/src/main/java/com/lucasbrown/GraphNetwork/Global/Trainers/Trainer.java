package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map.Entry;
import java.util.stream.Collectors;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Outcome;
import com.lucasbrown.GraphNetwork.Local.Nodes.INode;
import com.lucasbrown.GraphNetwork.Local.Nodes.IOutputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.ITrainable;
import com.lucasbrown.GraphNetwork.Local.Nodes.OutputNode;
import com.lucasbrown.NetworkTraining.History;
import com.lucasbrown.NetworkTraining.ApproximationTools.ErrorFunction;
import com.lucasbrown.NetworkTraining.ApproximationTools.WeightedAverage;
import com.lucasbrown.NetworkTraining.DataSetTraining.IFilter;

import jsat.linear.DenseVector;
import jsat.linear.Vec;

public class Trainer implements ITrainer{

    private final GraphNetwork network;

    public final WeightsLinearizer weightLinearizer;
    public final FilterLinearizer filterLinearizer;
    public final NetworkInputEvaluater networkEvaluater;
    public final IGradient weightsGradient;
    public final IGradient probabilityGradient;
    public final ISolver weightsSolver;
    public final ISolver probabilitySolver;

    private Vec weightsDeltas;
    private Vec probabilityDeltas;

    protected Double[][][] inputs;
    protected Double[][][] targets;

    public Trainer(NetworkInputEvaluater networkEvaluater, IGradient weightsGradient, ISolver weightsSolver, IGradient probabilityGradient, ISolver probabilitySolver, WeightsLinearizer weightLinearizer, FilterLinearizer filterLinearizer) {
        
        this.weightsGradient = weightsGradient;
        this.probabilityGradient = probabilityGradient;
        this.weightsSolver = weightsSolver;
        this.probabilitySolver = probabilitySolver;

        this.weightLinearizer = weightLinearizer;
        this.filterLinearizer = filterLinearizer;
        this.networkEvaluater = networkEvaluater;

        network = networkEvaluater.network;
    }

    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][] inputs, Double[][] targets) {
        setTrainingData(new Double[][][]{inputs}, new Double[][][]{targets});
    }

    
    /**
     * input and target dimension : [timestep][node]
     * 
     * @param inputs
     * @param targets
     */
    public void setTrainingData(Double[][][] inputs, Double[][][] targets) {
        this.inputs = inputs;
        this.targets = targets;
    }

    public void trainNetwork(int steps, int print_interval) {
        while (steps-- > 0) {
            trainingStep(steps % print_interval == 0);
        }
    }

    public void trainingStep(boolean print_forward) {
        History<Outcome, INode>[] histories = computeAllHistories();
        if(print_forward){
            printNetwork(histories);
        }

        Vec weightsGradient = aggregateWeightGradients(histories);
        Vec probabilityGradient = aggregateProbabilityGradients(histories);

        weightsDeltas = weightsSolver.solve(weightsGradient);
        probabilityDeltas = probabilitySolver.solve(probabilityGradient);

        applyWeightDeltas();
        applyProbabilityDeltas();
    }

    public History<Outcome, INode>[] computeAllHistories(){
        History<Outcome, INode>[] histories = new History[inputs.length];
        for(int i = 0; i < inputs.length; i++){
            networkEvaluater.setInputData(inputs[i]);
            histories[i] = networkEvaluater.computeNetworkInference();
        }
        return histories;
    }    

    private Vec aggregateWeightGradients(History<Outcome, INode>[] histories){
        Vec gradient = new DenseVector(weightLinearizer.totalNumOfVariables);
        for(int i = 0; i < inputs.length; i++){
            weightsGradient.setTargets(targets[i]);
            Vec grad = weightsGradient.computeGradient(histories[i]);
            gradient.mutableAdd(grad);
        }
        return gradient.divide(inputs.length);
    }

    private Vec aggregateProbabilityGradients(History<Outcome, INode>[] histories){
        Vec gradient = new DenseVector(filterLinearizer.totalNumOfVariables);
        for(int i = 0; i < inputs.length; i++){
            probabilityGradient.setTargets(targets[i]);
            gradient.mutableAdd(probabilityGradient.computeGradient(histories[i]));
        }
        return gradient.divide(inputs.length);
    }

    private void printNetwork(History<Outcome, INode>[] histories) {
        History<Outcome, INode> history = histories[0]; // print an example
        int time_count = history.getNumberOfTimesteps();

        StringBuilder sb = new StringBuilder();
        ArrayList<OutputNode> nodes = network.getOutputNodes();

        for (int t = 0; t < time_count; t++) {
            sb.append("Time Step ");
            sb.append(t);
            sb.append("\n\t");

            for (int i = 0; i < nodes.size(); i++) {
                OutputNode node = nodes.get(i);
                ArrayList<Outcome> outcomes = history.getStateOfRecord(t, node);

                if(outcomes == null || outcomes.isEmpty()){
                    continue;
                }

                sb.append(node.getName());
                sb.append(": [");
                sb.append(outcomes.stream()
                        .sorted(Outcome::descendingProbabilitiesComparator)
                        .limit(2)
                        .map(Object::toString)
                        .collect(Collectors.joining(",")));
                sb.append("] | target = ");
                sb.append(targets[0][t][i]);
                sb.append("\n\t");
            }
            sb.append("\n");
        }

        sb.append("Accuracy error : ");
        sb.append(getTotalError(histories, weightsGradient));
        sb.append("\nConsistency error : ");
        probabilityGradient.setTargets(targets[0]);
        sb.append(getTotalError(histories, probabilityGradient));
        sb.append("\n");
        System.out.println(sb.toString());
    }

    public double getTotalError(History<Outcome, INode>[] histories, IGradient errorEvaluator){
        double error = 0;
        for(int i = 0 ; i < histories.length ; i++){
            History<Outcome, INode> history = histories[i];
            errorEvaluator.setTargets(targets[i]);
            error += errorEvaluator.getTotalError(history);
        }
        return error / histories.length;
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

    public static Trainer getDefaultTrainer(GraphNetwork network){
        WeightsLinearizer weightLinearizer = new WeightsLinearizer(network);
        FilterLinearizer filterLinearizer = new FilterLinearizer(network);
        NetworkInputEvaluater networkEvaluater = new NetworkInputEvaluater(network);

        ErrorFunction erf = new ErrorFunction.AugmentedRelativeError();

        DirectNetworkGradient netGradient = new DirectNetworkGradient(network, new ForwardNetworkGradient(weightLinearizer), null, erf, weightLinearizer.totalNumOfVariables); 
        ADAMSolver weightsSolver = new ADAMSolver(netGradient, weightLinearizer.totalNumOfVariables); 

        OutcomeChanceFilterGradient filterGradient = new OutcomeChanceFilterGradient(network, new ForwardFilterGradient(filterLinearizer), null, erf, filterLinearizer.totalNumOfVariables);
        ADAMSolver filterSolver = new ADAMSolver(filterGradient, filterLinearizer.totalNumOfVariables);

        return new Trainer(networkEvaluater, netGradient, weightsSolver, filterGradient, filterSolver, weightLinearizer, filterLinearizer);
    }
}
