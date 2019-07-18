package xyz.onerous.modularnetwork.layer.activationlayer;

import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.type.ActivationType;

public class SigmoidLayer extends ActivationLayer {

	public SigmoidLayer(int neurons, int pNeurons) {
		super(ActivationType.Sigmoid, neurons, pNeurons);
		
	}
	
	public SigmoidLayer(int neurons, int pNeurons, double[] bias, double[][] weight) throws InvalidInputLengthException {
		super(ActivationType.Sigmoid, neurons, pNeurons, bias, weight);
	}
	
	public double activationFunction(double weightedInput) {
		return 1.0 / (1.0 + Math.exp(-weightedInput));
	}
	
	public double activationPrimeFunction(double weightedInput) {
		double activationValue = activationFunction(weightedInput);
		
		return activationValue * (1.0 - activationValue);
	}
}
