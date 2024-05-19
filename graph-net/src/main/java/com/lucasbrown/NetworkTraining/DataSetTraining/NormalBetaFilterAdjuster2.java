package com.lucasbrown.NetworkTraining.DataSetTraining;

public class NormalBetaFilterAdjuster2 extends NormalBetaFilterAdjuster {

    private static double ACCURACY = 1E-12;

    public NormalBetaFilterAdjuster2(NormalBetaFilter filter, NormalDistribution nodeDistribution,
            BetaDistribution arcDistribution) {
        super(filter, nodeDistribution, arcDistribution, false);
    }



    @Override
    public double getExpectedValueOfLogLikelihood(double shift, double scale) {
        double variance_x = nodeDistribution.getVariance();
        final double w = (mean - nodeDistribution.getMean() + shift) / (root_2 * scale * variance);
        double eta = scale * variance / variance_x;
        eta *= eta;
        double alpha = arcDistribution.getAlpha();
        double beta = arcDistribution.getBeta();

        double A = getA(w, eta, variance_x);
        double B = getB(w, eta, variance_x);

        return root_2*scale*variance * (A * alpha + B * beta) / (alpha + beta);
    }


    public double getA(double w, double eta, double sigma_x) {
        return -(Math.log(2*Math.PI*sigma_x*sigma_x) + 1/eta + 2*w*w + 1)/(2 * Math.sqrt(2*eta*sigma_x*sigma_x));
    }

    /**
     * Compute the integral: \int_{-\infty}^{\infty}\ln\left(1-\frac{1}{\sqrt{2\pi\sigma_{x}^{2}}}e^{-\eta\left(x+w\right)^{2}}e^{-x^{2}}\right)\frac{1}{\sqrt{2\pi\sigma_{x}^{2}}}e^{-\eta\left(x+w\right)^{2}}dx
     * Using a finite series approximation: -\sum_{n=2}^{N+1}\frac{1}{n-1}\left(\frac{1}{\sqrt{2\pi\sigma_{x}^{2}}}\right)^{n}e^{-\left(1-\frac{\eta n}{\eta n+n-1}\right)\eta nw^{2}}\sqrt{\frac{\pi}{\eta n+n-1}}
     */
    public double getB(double w, double eta, double sigma_x) {

        double sum = 0;
        double delta;
        int n = 2;
        do{
            double n_eta = n * eta;
            double denom = n_eta + n - 1;
            double exponent = (n_eta/denom - 1)*n_eta*w*w;
            delta = Math.exp(exponent);
            delta *= Math.pow(root_2pi*sigma_x, -n);
            delta *= Math.sqrt(Math.PI / denom);
            delta /= n - 1;
            sum += delta;
            n++;
            assert Double.isFinite(delta);
            assert n < 100000;
        }while(delta > ACCURACY);

        return -sum;
    }


}
