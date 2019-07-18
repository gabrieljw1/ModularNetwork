package xyz.onerous.modularnetwork.layer.activationlayer;

import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.type.ActivationType;

public class LinearLayer extends ActivationLayer {
	public static final double CONSTANT_OF_PROPORTIONALITY = 1.0;

	public LinearLayer(int neurons, int pNeurons) {
		super(ActivationType.Linear, neurons, pNeurons);
		
	}
	
	public LinearLayer(int neurons, int pNeurons, double[] bias, double[][] weight) throws InvalidInputLengthException {
		super(ActivationType.Linear, neurons, pNeurons, bias, weight);
	}
	
	public double activationFunction(double weightedInput) {
		return CONSTANT_OF_PROPORTIONALITY * weightedInput;
	}
	
	public double activationPrimeFunction(double weightedInput) {
		return CONSTANT_OF_PROPORTIONALITY;
	}
}
