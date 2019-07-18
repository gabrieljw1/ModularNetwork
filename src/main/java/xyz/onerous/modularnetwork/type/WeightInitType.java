package xyz.onerous.modularnetwork.type;

public enum WeightInitType {
	Kaiming, //Used for ReLU
	Xavier, //Used for TanH() and softsign. [-1, 1] functions pretty much
	RandomNormal; //Not a good idea, but you can do it.
}
