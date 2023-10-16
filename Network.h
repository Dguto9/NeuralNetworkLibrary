#pragma once
class Matrix;
class Vector;

class Network
{
public:
	Vector* layers;
	Matrix* weights;
	Vector* biases;
	int layerCount;
	Network(int layers, int* sizes);
	~Network();
	Vector compute(const Vector& inputs);
	void initialize();
	void saveState(const char* path);
};

