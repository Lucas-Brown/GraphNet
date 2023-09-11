package src.Tests;

import java.util.HashSet;
import java.util.function.Consumer;

import src.GraphNetwork.GraphNetwork;
import src.GraphNetwork.Node;
import src.GraphNetwork.NormalTransferFunction;

/**
 * Test for a graph network alternating between 0 and 1 
 */
public class SwitchNet
{

    public static void main(String[] args)
    {
        GraphNetwork net = new GraphNetwork();

        Node n1 = new Node();
        Node n2 = new Node();
        net.nodes.add(n1);
        net.nodes.add(n2);

        n1.AddNewConnection(n2, new NormalTransferFunction(0f, 1f, 0.5f));
        n2.AddNewConnection(n1, new NormalTransferFunction(0f, 1f, 0.5f));
        
        net.corrector = new Alternator(n2);

        for(int i = 0; i < 10; i++)
        {
            net.Step();
            System.out.println("node 1: " + n1.ToVisualString() + "\tnode 2: " + n2.ToVisualString());
        }

    }

    static class Alternator implements Consumer<HashSet<Node>>{
        private boolean state;
        private Node alternatingNode;

        public Alternator(Node alternatingNode)
        {
            state = false;
            this.alternatingNode = alternatingNode;
        }
        
        @Override
        public void accept(HashSet<Node> nodes) {
            alternatingNode.SetNodeSignal(state ? 0 : 1); 
            state = !state;
            nodes.add(alternatingNode);
        }

    }
}