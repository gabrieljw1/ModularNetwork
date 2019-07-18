package xyz.onerous.modularnetwork.learner;

import xyz.onerous.modularnetwork.ModularNetwork;
import xyz.onerous.modularnetwork.datapackage.WeightBiasDeltaPackage;
import xyz.onerous.modularnetwork.type.LossType;

public interface BatchTrainer {
	public WeightBiasDeltaPackage performBatch(ModularNetwork modularNetwork, double[][] batchData, int[] expectedOutputs, LossType lossType, boolean usingSoftmax, double learningRate);
}
