#include "Network.h"
#include "Vector.h"
#include "Matrix.h"
#include "Utils.h"
#include <stdlib.h>
#include <fstream>
#include <string>

Network::Network(int layerCount, int* sizes) {
	this->layerCount = layerCount;
	layers = (Vector*)calloc(this->layerCount, sizeof(Vector));
	biases = (Vector*)calloc(this->layerCount - 1, sizeof(Vector));
	weights = (Matrix*)calloc(this->layerCount - 1, sizeof(Matrix));
	for (int i = 0; i < layerCount; i++) {
		layers[i] = Vector(sizes[i]);
		if (i != layerCount - 1) {
			weights[i] = Matrix(sizes[i + 1], sizes[i]);
			biases[i] = Vector(sizes[i + 1]);
		}
	}
}

Network::~Network() {
	free(layers);
	free(weights);
	free(biases);
}

Vector Network::compute(const Vector& inputs) {
	layers[0] = inputs;
	layers[0].print();
	for (int i = 0; i < layerCount-1; i++) {
		layers[i + 1] = neuralNetUtils::sigmoid((weights[i] * layers[i]));
		//weights[i].print();
		//layers[i+1].print();
		//biases[i].print();
	}
	return layers[layerCount - 1];
}

void Network::initialize() {
	for (int i = 0; i < layerCount; i++) {
		layers[i].fill(0);
		if (i != layerCount - 1) {
			weights[i].randomize();
			biases[i].randomize();
		}
	}
}

void Network::saveState(const char* path) {
	std::string pathStr;
	for (int i = 0; i < layerCount - 1; i++) {
		pathStr = path;
		pathStr += "/weights-";
		pathStr += std::to_string(i);
		pathStr += ".ubyte-idx2";
		weights[i].writeToIDX(pathStr.c_str());
		pathStr = path;
		pathStr += "/biases-";
		pathStr += std::to_string(i);
		pathStr += ".ubyte-idx1";
		biases[i].writeToIDX(pathStr.c_str());
	}
}




