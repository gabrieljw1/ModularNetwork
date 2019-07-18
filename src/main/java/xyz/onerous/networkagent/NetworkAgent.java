package xyz.onerous.networkagent;

import xyz.onerous.modularnetwork.ModularNetwork;
import xyz.onerous.modularnetwork.exception.InvalidInputLengthException;
import xyz.onerous.modularnetwork.type.ActivationType;
import xyz.onerous.modularnetwork.type.LossType;
import xyz.onerous.modularnetwork.type.WeightInitType;

public class NetworkAgent {
	public static void main(String[] args) {
		ModularNetwork network = new ModularNetwork(2);
		
		network.addLayer(ActivationType.ReLU, 4);
		network.addLayer(ActivationType.Linear, 4);
		network.addLayer(ActivationType.ReLU, 4);
		network.addLayer(ActivationType.Sigmoid, 1);
		
		network.initializeWeightsDynamic();
		
	
		double[] networkOutput = new double[1];
		
		try {
			networkOutput = network.activateNetwork(new double[] {0.5, 0.9});
		} catch (InvalidInputLengthException e) {
			e.printStackTrace();
		}
		
		network.gradientDescent(0, LossType.CrossEntropy, true, 0.001);
		
		System.out.println(networkOutput[0]);
	}
}
