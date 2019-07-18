package xyz.onerous.modularnetwork.learner.multithread;

import xyz.onerous.modularnetwork.ModularNetwork;
import xyz.onerous.modularnetwork.datapackage.WeightBiasDeltaPackage;
import xyz.onerous.modularnetwork.learner.BatchTrainer;
import xyz.onerous.modularnetwork.type.LossType;

public class MTBatchTrainer extends Thread implements BatchTrainer {	
	private Thread t;
	private String threadName;
	
	protected WeightBiasDeltaPackage[] deltaPackages;
	public WeightBiasDeltaPackage totalDeltaPackage;
	
	private MTTrainThread[] trainThreads;
	
	private boolean finished = false;
	
	public MTBatchTrainer() {}
	
	public void run() {
		for (MTTrainThread thread : trainThreads) {
			thread.start();
		}

		deltaPackages = new WeightBiasDeltaPackage[trainThreads.length];
		
		for (int i = 0; i < trainThreads.length; i++) {
			while (trainThreads[i].deltaPackage == null) { 
				Thread.yield();
			}
			deltaPackages[i] = trainThreads[i].deltaPackage;
		}
		
		totalDeltaPackage = WeightBiasDeltaPackage.concatPackages(deltaPackages);	
		
		finished = true;
	}
	
	public void start() {
		if (t == null) {
			t = new Thread(this, threadName);
			t.start();
		}
	}
	
	public WeightBiasDeltaPackage performBatch(ModularNetwork modularNetwork, double[][] batchData, int[] expectedOutputs, LossType lossType, boolean usingSoftmax, double learningRate) {
		int batchSize = batchData.length;
		
		trainThreads = new MTTrainThread[batchSize];
		
		for (int i = 0; i < batchSize; i++) {
			trainThreads[i] = new MTTrainThread(modularNetwork, batchData[i], expectedOutputs[i], lossType, usingSoftmax, learningRate, "TrainThread-" + i);
		}
		
		threadName = "Batch Trainer";
		
		start();
		
		while (!finished) {
			Thread.yield();
		}
		
		return totalDeltaPackage;
	}
}
