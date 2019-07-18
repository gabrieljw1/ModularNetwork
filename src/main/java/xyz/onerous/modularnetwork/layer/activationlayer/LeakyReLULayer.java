package xyz.onerous.modularnetwork.layer.activationlayer;

import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.type.ActivationType;

public class LeakyReLULayer extends ActivationLayer {

	public LeakyReLULayer(int neurons, int pNeurons) {
		super(ActivationType.LeakyReLU, neurons, pNeurons);
		
	}
	
	public LeakyReLULayer(int neurons, int pNeurons, double[] bias, double[][] weight) throws InvalidInputLengthException {
		super(ActivationType.LeakyReLU, neurons, pNeurons, bias, weight);
	}
	
	public double activationFunction(double weightedInput) {
		if (weightedInput >= 0) {
			return weightedInput;
		} else {
			return 0.01 * weightedInput;
		}
	}
	
	public double activationPrimeFunction(double weightedInput) {
		if (weightedInput >= 0) {
			return 1.0;
		} else {
			return 0.01;
		}
	}
}
