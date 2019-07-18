package xyz.onerous.modularnetwork.layer.activationlayer;

import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.type.ActivationType;

public class ReLULayer extends ActivationLayer {

	public ReLULayer(int neurons, int pNeurons) {
		super(ActivationType.ReLU, neurons, pNeurons);
		
	}
	
	public ReLULayer(int neurons, int pNeurons, double[] bias, double[][] weight) throws InvalidInputLengthException {
		super(ActivationType.ReLU, neurons, pNeurons, bias, weight);
	}
	
	public double activationFunction(double weightedInput) {
		if (weightedInput >= 0) {
			return weightedInput;
		} else {
			return 0;
		}
	}
	
	public double activationPrimeFunction(double weightedInput) {
		if (weightedInput > 0) {
			return 1.0;
		} else if (weightedInput == 0) {
			return 0.5;
		} else {
			return 0.0;
		}
	}
}
