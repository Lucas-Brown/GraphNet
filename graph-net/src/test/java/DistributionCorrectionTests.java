import com.lucasbrown.GraphNetwork.Local.BellCurveDistribution;
import com.lucasbrown.GraphNetwork.Local.BellCurveDistributionAdjuster;

public class DistributionCorrectionTests {
    
    public static void main(String[] args)
    {
        BellCurveDistribution parent = new BellCurveDistribution(0, 1);
        BellCurveDistributionAdjuster adjuster = new BellCurveDistributionAdjuster(parent);

        for(int i = 0; i < 100; i++)
        {
            adjuster.addPoint(1, true, 1);
            adjuster.newtonUpdate();
            parent.setParamsFromAdjuster(adjuster);
        }

        System.out.println(adjuster.getMean());
        System.out.println(adjuster.getVariance());
    }
}
