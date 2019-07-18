package xyz.onerous.modularnetwork.layer.activationlayer;

import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.type.ActivationType;

public class TanHLayer extends ActivationLayer {

	public TanHLayer(int neurons, int pNeurons) {
		super(ActivationType.TanH, neurons, pNeurons);
		
	}
	
	public TanHLayer(int neurons, int pNeurons, double[] bias, double[][] weight) throws InvalidInputLengthException {
		super(ActivationType.TanH, neurons, pNeurons, bias, weight);
	}
	
	public double activationFunction(double weightedInput) {
		return Math.tanh(weightedInput);
	}
	
	public double activationPrimeFunction(double weightedInput) {
		return Math.pow(Math.cosh(weightedInput), -2);
	}
}
