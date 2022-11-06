#pragma once

#include <math.h>
#include <vector>

using namespace std;

using vect = vector<double>;
using vect2 = vector<vector<double>>;

// layer of neurons. store neurons and can calcForward and back
class Neurocolumn
{

//	vector<Neuron*> neu; // vector of neurons

	vect _a; // activity of 
	vect _b; // offset b in function z = b + w * a
	vect _db; // correction of b term
	vect _exp; //  expected value for calc of error
	vect2 _w; // weights of every neurons of prev layer
	vect2 _dw; // correction of w

	int _size; // count of neurons in this column
	int _sizePrev;  // count of neurons in previous column (or in-layer)

public:
	// n is count of neurons in this layer and nPrev - in previous layer
	Neurocolumn(int n, int nPrev); // initialization

	// c is factor of correction (related with learning rate)
	void correctAll(double c); // correct neurons

	// prev - is activation of prev layer
	void calcActivation(vect& prev); // forvard calculations for first layer
	void calcActivation(Neurocolumn* prev); // and other layer

	void calcBack(vect& prev); // back propagation from last (first of layers)
	void calcBack(Neurocolumn* prev); // back propagation in middle of net
	void setExpected(vect& expected); // set vector of true answers (when training)
	double calcError(vect& out); // calc difference between true answers and answer of neurocolumn
};


class Neuronet
{
public:

	double learnRate = 0.01; // learn rate for changing speed of learning (if big then less speed but more accurancy)
	double errorPack; // store sum of error in learning pack
	double errorOne;        // store error in one step forward-back
	double errorTotal;        // store sum of error in all learning set
	vector<Neurocolumn*> column; // layers of neurons (input is not here)

	// cols is vector of neurons in all layers vector {3,4,4,4,5} - 3 input neurons, 4x3 internal layers, 5 output neurons
	void init(vector<int>& cols);

	// out is output data for training
	void setExpected(vect& out); // put expected values in last layer for back prop and error

	// c is coef that factor for correction
	void correctAll(double c); // correction w and b after miniset of forward-back steps 

	// in - is input data for test
	void calcForward(vect& in); // calculation forward and calc answers =)

	// in - is input data for test
	void calcBack(vect& in); // calculation back (expected values must be in last layer)

	// in - is input data for test
	// out is output data for test
	// one step forward-back for one in and out vector
	void calcForwardBack(vector<double>& in, vector<double>& out);

	// vector in is vector of input data
	// vector in is vector of input data 
	// set of steps using all data (in.size() forward-back steps) with n corrections
	void learnPack(vector<vector<double>>& in, vector<vector<double>>& out, int n);

	// vector in is vector of input data
	// vector in is vector of input data 
	// mixing vectors in and out for more uniform and homogeneous data
	void mixData(vector<vector<double>>& in, vector<vector<double>>& out);


};
