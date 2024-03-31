import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveDistribution;

public class DistributionConvergence {

    public static void main(String[] args) {
        BellCurveDistribution bcd = new BellCurveDistribution(0.2, 1, 1, 10);

        double checkpoint = 1;
        double checkpoint_factor = 1.5;
        int n_iter = 1000;
        ArrayList<Double> means = new ArrayList<Double>((int) Math.ceil(Math.log(n_iter)/Math.log(checkpoint_factor)));

        for (int i = 0; i < n_iter; i++) {
            bcd.prepareReinforcement(0);
            bcd.prepareReinforcement(1);
            bcd.prepareReinforcement(2);
            bcd.prepareReinforcement(3);
            bcd.applyAdjustments();

            if(i >= checkpoint)
            {
                checkpoint *= checkpoint_factor;
                means.add(bcd.getMean());
            }
        }

        System.out.println(means.toString());
        System.out.println();
    }
}
