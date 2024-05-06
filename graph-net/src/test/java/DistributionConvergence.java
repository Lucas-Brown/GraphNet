import java.util.ArrayList;

import com.lucasbrown.GraphNetwork.Distributions.BellCurveFilter;

public class DistributionConvergence {

    public static void main(String[] args) {
        BellCurveFilter bcd = new BellCurveFilter(0.2, 1, 10, 1000);

        double checkpoint = 1;
        double checkpoint_factor = 1.5;
        int n_iter = 1000;
        ArrayList<Double> means = new ArrayList<Double>((int) Math.ceil(Math.log(n_iter)/Math.log(checkpoint_factor)));

        for (int i = 0; i < n_iter; i++) {
            // bcd.prepareReinforcement(0, weightFunction(bcd, 0));
            // bcd.prepareReinforcement(1, weightFunction(bcd, 0));
            // bcd.prepareReinforcement(2, weightFunction(bcd, 0));
            // bcd.prepareReinforcement(3, weightFunction(bcd, 0));
            bcd.prepareWeightedReinforcement(0);
            bcd.prepareWeightedReinforcement(1);
            bcd.prepareWeightedReinforcement(2);
            bcd.prepareWeightedReinforcement(3);
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
