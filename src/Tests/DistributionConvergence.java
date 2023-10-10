package src.Tests;

import src.GraphNetwork.Local.BellCurveDistribution;

public class DistributionConvergence {
    
    public static void main(String[] args)
    {
        BellCurveDistribution bcd = new BellCurveDistribution(0, 1, 1000000);
        for(int i=0; i < 1000000; i++)
        {
            bcd.reinforceDistribution(0, 1);
            bcd.diminishDistribution(1, 0.1);
        }

    }
}
