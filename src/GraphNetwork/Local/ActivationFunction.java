package src.GraphNetwork.Local;

/**
 * An activation function allows traditional neural networks to be universal function approximators.
 * In graphs, this is not necessary since all functions can be represented as infinite series which can occur in graphs.
 * Using an activation function may still allow for better results in less time due to their nonlinearity. 
 * Activation functions must also be invertible for the backwards-null signal to propagate. 
 */
public interface ActivationFunction {
	
	public abstract double activator(double x);
	public abstract double derivative(double x);
	public abstract double inverse(double x);

    public static class Linear implements ActivationFunction{ 

        @Override
        public double activator(double x) {
            return x;
        }

        @Override
        public double derivative(double x) {
            return 1;
        }

        @Override
        public double inverse(double x) {
            return x;
        }
        
    }
    
    public static class Sigmoid implements ActivationFunction{ // (0, 1)
        
        @Override
        public double activator(double x) {
            return 1.0 / (1.0 + Math.exp(-x));
        }

        @Override
        public double derivative(double x) {
            double i = this.activator(x);
            return (i) * (1.0 - i);
        }

        @Override
        public double inverse(double x) {
            return Math.log(x/(1-x));
        }
        
    }
    
    public static class TanH implements ActivationFunction{ // (-1, 1)

        @Override
        public double activator(double x) {
            return Math.tanh(x);
        }

        @Override
        public double derivative(double x) {
            double i = this.activator(x);
            return 1 - i * i;
        }

        @Override
        public double inverse(double x) {
            return Math.log((1+x)/(1-x))/2;
        }
        
    }
    
    public static class ArcTan implements ActivationFunction{ // ( -pi/2, pi/2)
        
        @Override
        public double activator(double x) {
            return Math.atan(x);
        }

        @Override
        public double derivative(double x) {
            return 1.0 / (x * x + 1.0);
        }

        @Override
        public double inverse(double x) {
            return Math.tan(x);
        }
        
    }
    
    public static class SoftSign implements ActivationFunction{ // (-1, 1)
        
        @Override
        public double activator(double x) {
            return x / ( 1 + Math.abs(x));
        }

        @Override
        public double derivative(double x) {
            return 1.0 / Math.pow(1 + Math.abs(x), 2);
        }

        @Override
        public double inverse(double x) {
            return x / ( 1 - Math.abs(x));
        }
        
    }
	 
}