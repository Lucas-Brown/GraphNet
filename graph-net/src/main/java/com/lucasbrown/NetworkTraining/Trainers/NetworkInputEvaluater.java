package com.lucasbrown.NetworkTraining.Trainers;

import java.util.HashMap;

import com.lucasbrown.GraphNetwork.Global.GraphNetwork;
import com.lucasbrown.GraphNetwork.Local.Nodes.IInputNode;
import com.lucasbrown.GraphNetwork.Local.Nodes.InputNode;
import com.lucasbrown.NetworkTraining.History.NetworkHistory;

public class NetworkInputEvaluater {

    protected GraphNetwork network;
    protected NetworkHistory networkHistory;

    private int timestep;
    protected Double[][] inputs;

    public NetworkInputEvaluater(GraphNetwork network) {
        this.network = network;
        timestep = 0;

        network.setInputOperation(this::applyInputToNode);
        networkHistory = new NetworkHistory(network);
    }

    public NetworkInputEvaluater(GraphNetwork network, Double[][] inputs){
        this(network);
        setInputData(inputs);
    }

    public void setInputData(Double[][] inputs) {
        this.inputs = inputs;
    }

    public NetworkHistory computeNetworkInference() {
        if (inputs == null) {
            return null;
        }
        network.deactivateAll();
        networkHistory = new NetworkHistory(network);
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
