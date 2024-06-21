package com.lucasbrown;

import com.lucasbrown.GraphNetwork.Local.ActivationFunction;
import com.lucasbrown.GraphNetwork.Local.Outcome;

import jsat.linear.DenseMatrix;
import jsat.linear.LUPDecomposition;
import jsat.linear.Matrix;

public class Understanding {

    private static double sigma(double x){
        return x;
    }

    private static double sigmaPrime(double x){
        return 1;
    }

    private static double sigmaDoublePrime(double x){
        return 0;
    }
    
    public static void main(String[] args) {

        double w = 1.1;
        double b = 0;
        int n_itter = 100;
        double epsilon = 01;

        double input = 1;
        double[] outputs = {2,4,8,16,32};

        for(int itter = 0; itter < n_itter; itter++){
            Matrix paramJacobian = new DenseMatrix(2, 1);
            Matrix paramHessian = new DenseMatrix(2, 2); 
            Matrix errorJacobian = new DenseMatrix(2, 1);
            Matrix errorHessian = new DenseMatrix(2, 2); 

            double a_prev = input;
            double z, a = 0;
            for(double out : outputs){
                z = w*a_prev + b;
                a = sigma(z);
                System.out.println(a);

                Matrix z_jacobi = new DenseMatrix(2, 1);

                // construct the jacobian for the net value (z)
                // starting with the direct derivative of z
                z_jacobi.set(0, 0, a_prev);
                z_jacobi.set(1, 0, 1);

                // incorporate previous jacobians
                Matrix weighed_jacobi = paramJacobian.multiply(w);
                z_jacobi.mutableAdd(weighed_jacobi);

                // use the net value jacobian to compute the activation jacobian
                double activation_derivative = sigmaPrime(z);
                double activation_second_derivative = sigmaDoublePrime(z);

                // construct hessian
                Matrix JJT = z_jacobi.multiplyTranspose(z_jacobi);
                Matrix jacobi_chain = new DenseMatrix(2, 2);

                for (int j = 0; j < 2; j++) {
                    jacobi_chain.set(j, 0, paramJacobian.get(j, 0));
                }

                jacobi_chain.mutableAdd(jacobi_chain.transpose());
                jacobi_chain.mutableAdd(paramHessian.multiply(w));

                // finalize Hessian
                paramHessian.mutableAdd(JJT.multiply(activation_second_derivative));
                paramHessian.mutableAdd(jacobi_chain.multiply(activation_derivative));
                
                // apply to activated jacobi
                paramJacobian.mutableAdd(z_jacobi.multiply(activation_derivative));

                // // compute errors
                // double error_derivative = a-out;
                // double error_second_derivative = 1;

                // collect errors
                // errorJacobian.mutableAdd(paramJacobian.multiply(error_derivative));
                // errorHessian.mutableAdd(paramJacobian.multiplyTranspose(paramJacobian).multiply(error_second_derivative));
                // errorHessian.mutableAdd(paramHessian.multiply(error_derivative));

                a_prev = a;
            }
            System.out.println();

            
            double error_derivative = a-outputs[outputs.length - 1];
            double error_second_derivative = 1;
            
            errorJacobian = paramJacobian.multiply(error_derivative);
            errorHessian = paramJacobian.multiplyTranspose(paramJacobian).multiply(error_second_derivative);
            errorHessian.mutableAdd(paramHessian.multiply(error_derivative));

            Matrix G = errorJacobian.multiply(errorHessian.multiply(errorJacobian).transpose());

            Matrix eta = DenseMatrix.eye(2).multiply(epsilon);
            LUPDecomposition decomposition = new LUPDecomposition(G.add(eta));
            Matrix delta = decomposition.solve(errorJacobian);

            // for(int i = 0; i < 2; i++){
            //     delta.set(i, 0, Math.abs(delta.get(i, 0)* Math.signum(errorJacobian.get(i, 0))));
            // }

            w -= delta.get(0, 0);
            b -= delta.get(1, 0);
        }
    }
}
