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
			continue;
		}
		if (caption.c_str() == string("Iris-versicolor"))
		{
			out.back()[1] = 1.0;
			continue;
		}
		if (caption.c_str() == string("Iris-virginica"))
		{
			out.back()[2] = 1.0;
			continue;
		}
	}
	file.close();
	return true;
}

// sinus function in in-vector and out-vector (5 is points before testing point)
void loadSinus(vector<vector<double>>& in, vector<vector<double>>& out)
{
	int nn = 400;
	double dt = 0.2;
	double start = 0.0;
	for (int i = 0; i < nn; i++)
	{
		start = start + dt * i;
		in.push_back({});
		in.back().resize(5);
		in.back()[0] = 0.5 + 0.5 * sin(start);
		in.back()[1] = 0.5 + 0.5 * sin(start + dt);
		in.back()[2] = 0.5 + 0.5 * sin(start + dt * 2);
		in.back()[3] = 0.5 + 0.5 * sin(start + dt * 3);
		in.back()[4] = 0.5 + 0.5 * sin(start + dt * 4);
		out.push_back({});
		out.back().resize(1);
		out.back()[0] = 0.5 + 0.5 * sin(start + dt * 5);
	}
}


int main()
{
	// vector of neuro-columns in neuronet
	vector<int> config; // configuration of neuro columns (size config is number of neurocolumns and elements is number of neurons in every column  )
	vector<vector<double>> in; // input data
	vector<vector<double>> out; // output data (look description of function 'loadFisher')

	int typeTask = 1; // 0 - fisher's irises, 1 - sinus function
	Neuronet n;
	switch (typeTask)
	{
	case 0:
		if (!loadFisher(in, out))
		{
			cout << "no data - no deal!" << endl;
			return 1;
		}
		n.learnRate = 0.001;
		config.resize(4); // configuration 4-8-8-3
		config[0] = 4;
		config[1] = 8;
		config[2] = 8;
		config[3] = 3;
		break;
	case 1:
		n.learnRate = 0.01;
		loadSinus(in, out); 
		config.resize(3); // configuration 5-3-1
		config[0] = 5;
		config[1] = 3;
		config[2] = 1;
		break;
	}

	n.init(config);

	// it's learning of neuronet
	double t1 = omp_get_wtime();
	int nPack = 5; // number of every packs of data for smoothing
	int steps = 1000000; // max limit steps
	double limitError = 0.05; // max limit error
	int showInfoEverySteps = 1000;
	do
	{
		n.learnPack(in, out, nPack); // in (and out) mix and distribute by nPack (you can change this value) vectors for learning
		if(steps % showInfoEverySteps == 0)
			cout << "error = " << sqrt(n.errorTotal / in.size()) << endl; // sum of errors in all dataset = sqr(out - activation) where activation is activation in last column
		steps--;
	} while ((sqrt(n.errorTotal / in.size()) > limitError) && steps!=0);

	double t2 = omp_get_wtime();
	cout << "time = " << t2 - t1 << endl;
	system("pause"); // thank you for yor attention!
	return 0;
}