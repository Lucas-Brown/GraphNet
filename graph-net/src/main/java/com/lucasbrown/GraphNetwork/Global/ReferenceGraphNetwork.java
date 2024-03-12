package com.lucasbrown.GraphNetwork.Global;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.FilterDistribution;
import com.lucasbrown.GraphNetwork.Local.Node;
import com.lucasbrown.GraphNetwork.Local.Arc;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.InputReferenceNode;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.OutputReferenceNode;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.ReferenceArc;
import com.lucasbrown.GraphNetwork.Local.ReferenceStructure.ReferenceNode;

/**
 * A neural network using a probabalistic directed graph representation.
 * Training currently a work in progress
 * 
 * Current representation allows for both positive and negative reinforcement.
 * Only postive reinforcement is implemented currently.
 */
public class ReferenceGraphNetwork extends GraphNetwork{

    @Override
    public ReferenceNode getNewHiddenNode(final ActivationFunction activationFunction) {
        ReferenceNode n = new ReferenceNode(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    @Override
    public InputReferenceNode getNewInputNode(final ActivationFunction activationFunction) {
        InputReferenceNode n = new InputReferenceNode(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    @Override
    public OutputReferenceNode getNewOutputNode(final ActivationFunction activationFunction) {
        OutputReferenceNode n = new OutputReferenceNode(this, networkData, activationFunction);
        nodes.add(n);
        return n;
    }

    @Override
    public void addNewConnection(Node transmittingNode, Node recievingNode,
            FilterDistribution transferFunction) {
        // boolean doesConnectionExist =
        // transmittingNode.DoesContainConnection(recievingNode);
        // if(!doesConnectionExist)
        // {
        ReferenceArc connection = new ReferenceArc(transmittingNode, recievingNode, transferFunction);
        transmittingNode.addOutgoingConnection(connection);
        recievingNode.addIncomingConnection(connection);
        // }
        // return doesConnectionExist;
    }


    

    /**
     * Fill the parameters, weights, and biases arrays with the flattened data.
     * each to-fill array is expected to be the correct shape
     * 
     * @param data
     * @param parameters_to_fill
     * @param weights_to_fill
     * @param biases_to_fill
     */
    /*
    private void inflateNodeData(double[] data, double[][][] parameters_to_fill, double[][][] weights_to_fill,
            double[][] biases_to_fill) {
        int data_idx = 0;

        // input node data
        for (int i = 0; i < n_input; i++) {
            for (int j = 0; j < parameters_to_fill[i].length; j++) {
                for (int k = 0; k < parameters_to_fill[i][j].length; k++) {
                    parameters_to_fill[i][j][k] = data[data_idx++];
                }
            }
        }

        // hidden node data
        for (int i = 0; i < n_hidden; i++) {

            // params
            for (int j = 0; j < parameters_to_fill[i + n_input].length; j++) {
                for (int k = 0; k < parameters_to_fill[i + n_input][j].length; k++) {
                    parameters_to_fill[i + n_input][j][k] = data[data_idx++];
                }
            }

            // weights
            for (int j = 0; j < weights_to_fill[i].length; j++) {
                for (int k = 0; k < weights_to_fill[i][j].length; k++) {
                    weights_to_fill[i][j][k] = data[data_idx++];
                }
            }

            // biases
            for (int j = 0; j < biases_to_fill[i].length; j++) {
                biases_to_fill[i][j] = data[data_idx++];
            }
        }

    }
    */

    /**
     * Flatten the data by node, then by data type.
     * 
     * @param parameters
     * @param weights
     * @param biases
     * @return
     */
    /*
    private double[] flattenNodeData(double[][][] parameters, double[][][] weights, double[][] biases) {
        DoubleStream flatDataPerNode = DoubleStream.empty();

        // data for the parameters for input nodes
        for (int i = 0; i < n_input; i++) {
            DoubleStream params_i = Stream.of(parameters[i]).flatMapToDouble(d -> DoubleStream.of(d));
            DoubleStream.concat(flatDataPerNode, params_i);
        }

        // data for all hidden nodes
        for (int i = 0; i < parameters.length; i++) {
            // flatten each array
            DoubleStream params_i = Stream.of(parameters[i + n_input]).flatMapToDouble(d -> DoubleStream.of(d));
            DoubleStream weights_i = Stream.of(weights[i]).flatMapToDouble(d -> DoubleStream.of(d));
            DoubleStream biases_i = DoubleStream.of(biases[i]);

            // append node data
            DoubleStream.concat(flatDataPerNode, params_i);
            DoubleStream.concat(flatDataPerNode, weights_i);
            DoubleStream.concat(flatDataPerNode, biases_i);
        }

        // output nodes have no data

        return flatDataPerNode.toArray();
    }

    private void fillNodeData(boolean[][] connectivityMatrix, double[][][] parameters, double[][][] weights,
            double[][] biases, int[] node_indices) {
        final int n_exposed = n_input + n_output;

        int node_idx = 0;
        for (int i = 0; i < n_input; i++) {
            parameters[i] = nodes.get(i).getConnectionParameters();
            node_idx += parameters[i].length;
        }


        for (int i = n_exposed; i < connectivityMatrix.length; i++) {
            final int hidden_idx = i - n_exposed;
            node_indices[hidden_idx] = node_idx;

            Node node_i = nodes.get(i);
            connectivityMatrix[hidden_idx] = node_i.getConnectivity();
            double[][] params_i = parameters[hidden_idx + n_input] = node_i.getConnectionParameters();
            double[][] weights_i = weights[hidden_idx] = node_i.getWeights();
            double[] biases_i = biases[hidden_idx] = node_i.getBiases();

            // find the flattened index of the next node
            for (int j = 0; j < params_i.length; j++) {
                node_idx += params_i[j].length;
            }

            for (int j = 0; j < weights_i.length; j++) {
                node_idx += weights_i[j].length;
            }

            node_idx += biases_i.length;
        }
    }

    @Override
    public GeneticData getGeneticRepresentation() {

        final int n_total = nodes.size();

        boolean[][] connectivityMatrix = new boolean[n_total][n_total];
        double[][][] parameters = new double[n_hidden + n_input][][];
        double[][][] weights = new double[n_hidden][][];
        double[][] biases = new double[n_hidden][];
        int[] node_indices = new int[n_hidden];

        fillNodeData(connectivityMatrix, parameters, weights, biases, node_indices);
        double[] flattenedData = flattenNodeData(parameters, weights, biases);
        return new GeneticData(n_input, n_output, n_hidden, connectivityMatrix, node_indices, flattenedData);
    }

    @Override
    public GraphNetwork getNetworkFromGeneticData(GeneticData data, ActivationFunction activationFunction) {
        GraphNetwork net = new GraphNetwork();
        final int n_total = data.inputNodeCount + data.outputNodeCount + data.hiddenNodeCount;

        // create new input nodes
        for (int i = 0; i < data.inputNodeCount; i++) {
            net.createInputNode(activationFunction);
        }

        // create new output nodes
        for (int i = 0; i < data.outputNodeCount; i++) {
            net.createOutputNode(activationFunction);
        }

        // create new hidden nodes
        for (int i = 0; i < data.hiddenNodeCount; i++) {
            net.createHiddenNode(activationFunction);
        }

        // construct all the arcs/connectivities
        for (int i = 0; i < n_total; i++) {
            Node node_i = net.nodes.get(i);
            for (int j = 0; j < n_total; i++) {
                if (data.connectivityMatrix[i][j]) // if i is sending to j
                {
                    // THIS SHOULD NOT BE HARD CODED
                    BellCurveDistribution transferFunction = new BellCurveDistribution(0, 1);
                    net.addNewConnection(node_i, net.nodes.get(j), transferFunction);
                }
            }
        }

        // get the structure of the data and fill
        boolean[][] connectivityMatrix = new boolean[n_total][n_total];
        double[][][] parameters = new double[net.n_hidden + net.n_input][][];
        double[][][] weights = new double[net.n_hidden][][];
        double[][] biases = new double[net.n_hidden][];
        int[] node_indices = new int[net.n_hidden];

        net.fillNodeData(connectivityMatrix, parameters, weights, biases, node_indices);
        net.inflateNodeData(data.data, parameters, weights, biases);

        // set the data for each node
        for (int i = 0; i < net.n_input; i++) {
            net.nodes.get(i).setConnectionsData(parameters[i], new double[1][1], new double[1]);
        }
        for (int i = 0; i < net.n_hidden; i++) {
            net.nodes.get(i + net.n_input + net.n_output).setConnectionsData(
                    parameters[i + net.n_input], weights[i], biases[i]);
        }

        return net;
    }
    */

}
