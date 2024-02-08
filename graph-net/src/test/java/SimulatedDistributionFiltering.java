

import java.util.Random;

import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;


public class SimulatedDistributionFiltering {
    
    public static void main(String[] args)
    {
        Random rng = new Random();
        BellCurveDistribution prior_prob = new BellCurveDistribution(1, 0.5);
        double p_n1 = prior_prob.computeNormalizedDist(-1);
        double p_0 = prior_prob.computeNormalizedDist(0);
        double p_1 = prior_prob.computeNormalizedDist(1);

        BellCurveDistribution bcd = new BellCurveDistribution(0, 1);
        for(int i=0; i < 10000; i++)
        {
            if(rng.nextDouble() < p_0)  bcd.reinforceDistribution(0, p_0);
            if(rng.nextDouble() < p_n1) bcd.reinforceDistribution(-1, p_n1);
            if(rng.nextDouble() < p_1)  bcd.diminishDistribution(1, p_1);
        }

        System.out.println();

    }
}
