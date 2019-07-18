package xyz.onerous.modularnetwork.datapackage;

import xyz.onerous.modularnetwork.util.MatrixUtil;

public class WeightBiasDeltaPackage {
	public double[][][] deltaW;
	public double[][]   deltaB;
	
	public WeightBiasDeltaPackage(double[][][] deltaW, double[][] deltaB) {
		this.deltaW = deltaW;
		this.deltaB = deltaB;
	}
	
	public WeightBiasDeltaPackage(WeightBiasDeltaPackage deltaPackage) {
		this.deltaW = deltaPackage.deltaW;
		this.deltaB = deltaPackage.deltaB;
	}
	
	/**
	 * @param deltaPackages an array of WeightBiasDeltaPackages
	 * @return the sum of all delta packages passed in inside of one single delta package.
	 */
	public static WeightBiasDeltaPackage concatPackages(WeightBiasDeltaPackage[] deltaPackages) {
		if (deltaPackages.length == 0) {
			return (WeightBiasDeltaPackage) null;
		}
		
		WeightBiasDeltaPackage output = new WeightBiasDeltaPackage(deltaPackages[0]);
		
		for (int p = 1; p < deltaPackages.length; p++) {
			output.deltaW = MatrixUtil.add(output.deltaW, deltaPackages[p].deltaW);
			output.deltaB = MatrixUtil.add(output.deltaB, deltaPackages[p].deltaB);
		}
		
		return output;
	}
}
