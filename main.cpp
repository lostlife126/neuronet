#include "main.h"


///// loading 'iris.data' from .... to vector of vector of parameters
// in is vector of 4 doubles (sepal_len, sepal_wit, petal_len, petal_wit)
// out is vector of irises (1,0,0) (0,1,0) (0,0,1)

bool loadFisher(vector<vector<double>>& in, vector<vector<double>>& out)
{
	fstream file("iris.data");
	if (!file.is_open())
	{
		cout << "File 'iris.data' is not found!" << endl;
		return false;
	}

	// for simplifying i changed all ',' to ' ' in my iris.dat
	while (!file.eof())
	{
		in.push_back({});
		in.back().resize(4);
		out.push_back({});
		out.back().resize(3);
		file >> in.back()[0];
		file >> in.back()[1];
		file >> in.back()[2];
		file >> in.back()[3];
		out.back()[0] = 0.0;
		out.back()[1] = 0.0;
		out.back()[2] = 0.0;
		string caption;
		file >> caption;
		// there is no protect from mistakes in file!!!
		if (caption.c_str() == string("Iris-setosa"))
		{
			out.back()[0] = 1.0;
		}
		if (caption.c_str() == string("Iris-versicolor"))
		{
			out.back()[1] = 1.0;
		}
		if (caption.c_str() == string("Iris-virginica"))
		{
			out.back()[2] = 1.0;
		}
	}
	file.close();
	return true;
}


int main()
{
	// vector of neuro-columns in neuronet
	vector<int> par{4,8,8,3};// 4-in, 3-out, and 2x8 internal columns

	Neuronet n; 
	n.init(par); 

	vector<vector<double>> in; // input data
	vector<vector<double>> out; // output data (look description of function 'loadFisher')

	if (!loadFisher(in, out))
	{
		cout << "no data - no deal!"<<endl;
		return 1;
	}

	// it's learning of neuronet
	for (int i = 0; i < 1000; i++)
	{
		n.learn(in, out, 10); // in (and out) mix and distribute by 10 (you can change this value) vectors for learning
		cout << "error = " << n.total_error << endl; // sum of errors in all dataset = sqr(out - activation) where activation is activation in last column
	}

	system("pause"); // thank you for yor attention!
	return 0;
}