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
public class SwitchNet
{

    static int post_fire_count = 0;

    public static void main(String[] args)
    {
        GraphNetwork net = new GraphNetwork();

        Node n1 = new Node();
        Node n2 = new Node();
        net.nodes.add(n1);
        net.nodes.add(n2);

        net.AddNewConnection(n1, n2, new NormalTransferFunction(0f, 1f, 0.5f));
        net.AddNewConnection(n2, n1, new NormalTransferFunction(0f, 1f, 0.5f));
        
        net.corrector = new Alternator(n2);

        for(int i = 0; i < 1000; i++)
        {
            net.Step();
        }

        System.out.println("\nSIGNAL STOP\n");
        
        net.corrector = SwitchNet::PrintAllActiveNodes;
        for(int i = 0; i < 1000; i++)
        {
            net.Step();
        }

        System.out.println("steps before auto-stop: " + post_fire_count);
    }

    public static void PrintAllActiveNodes(HashSet<Node> signaledNodes)
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

    static class Alternator implements Consumer<HashSet<Node>>{
        private boolean state;
        private Node alternatingNode;

        public Alternator(Node alternatingNode)
        {
            state = false;
            this.alternatingNode = alternatingNode;
        }
        
        @Override
        public void accept(HashSet<Node> signaledNodes) {
            if(state = !state)
            {
                alternatingNode.SetNodeSignal(signaledNodes, 1); 
            }
        }

    }
}