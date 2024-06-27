package com.lucasbrown.GraphNetwork.Global.Trainers;

import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.Network.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.NetworkTraining.History;

public class NetworkInputEvaluater {

    protected GraphNetwork network;
    protected final History networkHistory;

    private int timestep;
    protected Double[][] inputs;

    public NetworkInputEvaluater(GraphNetwork network) {
        this.network = network;
        timestep = 0;

        network.setInputOperation(this::applyInputToNode);
        networkHistory = new History(network);
    }

    public void setInputData(Double[][] inputs) {
        this.inputs = inputs;
    }

    public History computeNetworkInference() {
        if (inputs == null) {
            return null;
        }
        networkHistory.burnHistory();
        captureForward();
        return networkHistory;
    }

    private void captureForward() {
        for (timestep = 0; timestep < inputs.length; timestep++) {
            network.trainingStep();
            // if (print_forward) {
            // System.out.println(network.toString() + " | Target = " +
            // Arrays.toString(targets[timestep]));
            // }
            networkHistory.captureState();
        }
    }

    private void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap) {
        applyInputToNode(inputNodeMap, inputs, timestep);
    }

    private static void applyInputToNode(HashMap<Integer, ? extends IInputNode> inputNodeMap, Double[][] input,
            int counter) {
        InputNode[] sortedNodes = inputNodeMap.values().stream().sorted().toArray(InputNode[]::new);

        for (int i = 0; i < sortedNodes.length; i++) {
            if (input[counter][i] != null) {
                sortedNodes[i].acceptUserForwardSignal(input[counter][i]);
            }
        }
    }

}
