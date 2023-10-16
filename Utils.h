#pragma once
class Vector;

namespace neuralNetUtils {
	float sigmoid(float x);
	Vector sigmoid(const Vector& x);
	float d_sigmoid(float x);
	int reverseInt(int x);
	int loadFromIDX(const char* path, int index);
	float meanSquaredError(const Vector x, const Vector y);
}