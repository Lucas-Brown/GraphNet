package src.Tests;

import java.util.HashSet;
import java.util.SortedSet;
import java.util.TreeSet;
import java.util.function.Consumer;

import src.GraphNetwork.GraphNetwork;
import src.GraphNetwork.Node;
import src.GraphNetwork.NormalTransferFunction;

/**
 * Test for a graph network alternating between 0 and 1 
 */
public class Alternating
{

    private static int post_fire_count = 0;

    public static void main(String[] args)
    {
        GraphNetwork net = new GraphNetwork();

        Node n1 = new Node();
        Node n2 = new Node();
        Node n3 = new Node();
        net.nodes.add(n1);
        net.nodes.add(n2);
        net.nodes.add(n3);


        //net.AddNewConnection(n1, n1, new NormalTransferFunction(0.1f, 1f, 0.4f));
        //net.AddNewConnection(n1, n1, new NormalTransferFunction(0.9f, 1f, 0.6f));
        
        net.AddNewConnection(n1, n2, new NormalTransferFunction(0.9f, 1f, 0.1f));
        net.AddNewConnection(n2, n1, new NormalTransferFunction(0.8f, 1f, 0.2f));
        net.AddNewConnection(n1, n3, new NormalTransferFunction(0.7f, 1f, 0.3f));
        net.AddNewConnection(n3, n1, new NormalTransferFunction(0.6f, 1f, 0.4f));
        //net.AddNewConnection(n3, n2, new NormalTransferFunction(0.5f, 1f, 0.5f));
        //net.AddNewConnection(n2, n3, new NormalTransferFunction(0.4f, 1f, 0.6f));
        
        net.corrector = new Alternator(n1);

        for(int i = 0; i < 100000; i++)
        {
            net.Step();
        }

        System.out.println("\nSIGNAL STOP\n");
        
        net.corrector = Alternating::PrintAllActiveNodes;
        for(int i = 0; i < 1000; i++)
        {
            net.Step();
        }

        System.out.println("steps before auto-stop: " + post_fire_count);
    }

    private static void PrintAllActiveNodes(HashSet<Node> signaledNodes)
    {
        TreeSet<Node> sSet = new TreeSet<Node>(signaledNodes);
        StringBuilder sb = new StringBuilder();
        sSet.forEach(node -> 
        {
            sb.append(node.toString());
            sb.append('\t');
        });
        String s = sb.toString();
        if(!s.trim().isEmpty())
        {
            post_fire_count++;
            System.out.println(s);
        }
    }

    private static class Alternator implements Consumer<HashSet<Node>>{
        private boolean state;
        private Node alternatingNode;

        public Alternator(Node alternatingNode)
        {
            state = true;
            this.alternatingNode = alternatingNode;
        }
        
        @Override
        public void accept(HashSet<Node> signaledNodes) {
            state = !state;
            alternatingNode.SetNodeSignal(signaledNodes, state ? 1 : 0); 
        }

    }
}