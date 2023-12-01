package src.Tests;

import src.GraphNetwork.Local.BellCurveDistribution;

public class DistributionConvergence {
    
    public static void main(String[] args)
    {
        BellCurveDistribution bcd = new BellCurveDistribution(0, 1);
        for(int i=0; i < 10000; i++)
        {
            bcd.reinforceDistribution(0);
            bcd.reinforceDistribution(-1);
            bcd.diminishDistribution(1);
        }

        System.out.println();

    }
}
