package src.Tests;

import src.GraphNetwork.Local.BellCurveDistribution;

public class DistributionConvergence {
    
    public static void main(String[] args)
    {
        BellCurveDistribution bcd = new BellCurveDistribution(0, 1);
        for(int i=0; i < 10000; i++)
        {
            bcd.reinforceDistributionNoFilter(0);
            bcd.reinforceDistributionNoFilter(-1);
            bcd.diminishDistributionNoFilter(1);
        }

        System.out.println();

    }
}
