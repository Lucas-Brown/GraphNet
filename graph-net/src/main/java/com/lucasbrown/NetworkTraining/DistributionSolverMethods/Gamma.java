package com.lucasbrown.NetworkTraining.DistributionSolverMethods;

public class Gamma {
    
        public static double gamma(double x){
            if(x >= 0){
                return sterling(x)/x;
            }
            else{
                return Math.PI/(Math.sin(Math.PI*x) * gamma(1-x));
            }
        }

        public static double sterling(double x)
        {
            return Math.sqrt(2*Math.PI*x)*Math.pow(x/Math.E, x)*(1 + 1/(12 * x));
        }
}
