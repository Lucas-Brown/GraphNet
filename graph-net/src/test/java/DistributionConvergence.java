import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;

public class DistributionConvergence {

    public static void main(String[] args) {
        BellCurveDistribution bcd = new BellCurveDistribution(0.2, 1);

        ArrayList<Double> means = new ArrayList<Double>(100);
        double checkpoint = 1;
        double checkpoint_factor = 2;

        for (int i = 0; i < 1000; i++) {
            bcd.prepareReinforcement(0);
            bcd.prepareReinforcement(1);
            bcd.applyAdjustments();

            if(i >= checkpoint)
            {
                checkpoint *= checkpoint_factor;
                means.add(bcd.getMean());
            }
        }

        System.out.println(means.toString());

    }
}
