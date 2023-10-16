#include "Utils.h"
#include "Vector.h"
#include <fstream>
#include <cassert>
#include <cmath>

float neuralNetUtils::sigmoid(float x) {
	return 1 / (1 + exp(-x));
}

Vector neuralNetUtils::sigmoid(const Vector& x) {
	Vector out = x;
	out.fill(0);
	for (int i = 0; i < x.dimension; i++) {
		out.data[i] = sigmoid(x.data[i]);
	}
	return out;
}

float neuralNetUtils::d_sigmoid(float x) {
	return (1 / -(1 + exp(-x))) * (1 / -(1 + exp(-x))) * -exp(-x);
}

int neuralNetUtils::reverseInt(int x) {
	int val = (x << 16) | ((x >> 16) & 0xFFFF);
	return ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
}

int neuralNetUtils::loadFromIDX(const char* path, int index) {
	int fileDimension;
	int tmpCount;
	int count = 1;
	unsigned char temp;
	std::ifstream file(path, std::ios::binary);
	if (file.is_open()) {
		file.read((char*)&fileDimension, sizeof(fileDimension));
		fileDimension = fileDimension >> 24;
		for (int i = 0; i < fileDimension; i++) {
			file.read((char*)&tmpCount, sizeof(tmpCount));
			tmpCount = reverseInt(tmpCount);
			count *= tmpCount;
		}
		assert(index <= count);
		file.seekg((index) + 4 + (4 * fileDimension), file.beg);
			file.read((char*)&temp, sizeof(temp));
			return (float)temp;
	}
	else {
		throw std::runtime_error("File not found");
	}
}

float neuralNetUtils::meanSquaredError(const Vector x, const Vector y) {
	assert(x.dimension == y.dimension);
	float out = 0;
	for (int i = 0; i < x.dimension; i++) {
		out += pow(x.data[i] - y.data[i], 2);
	}
	return out / (float)x.dimension;
}
