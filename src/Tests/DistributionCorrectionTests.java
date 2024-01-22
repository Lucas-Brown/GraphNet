package src.Tests;

import src.GraphNetwork.Local.BellCurveDistribution;
import src.GraphNetwork.Local.BellCurveDistributionAdjuster;

public class DistributionCorrectionTests {
    
    public static void main(String[] args)
    {
        BellCurveDistribution parent = new BellCurveDistribution(0, 1);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent);
        System.out.println("Completed pre-computations");

        for(int i = 0; i < 100; i++)
        {
            adjuster.addPoint(1, true, 1);
            adjuster.newtonUpdate();
        }

        System.out.println(adjuster.getMean());
        System.out.println(adjuster.getVariance());
    }
}
