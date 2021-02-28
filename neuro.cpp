#include "neuro.h"

//there i use logical sigma 
double sigma(double x)
{
	return 1.0 / (1.0 + exp(-x));
}

// derivative of sigma
double dsigma(double x)
{
	return x * (1.0 - x);
}


// random function (-1,+1)
double rand1()
{
	return (rand() % 32768) / 16384.0 - 1.0;
}

// sqr
double sqr(double x)
{
	return x * x;
}

///////////////////////////////////////////////////////////////////////////////


void Neuron::init(int nPrev)
{
	w.resize(nPrev);
	dw.resize(nPrev, 0.0);
	db = 0.0;
	exp = 0.0;
	a = 0.0;
	er = 0.0;
	b = rand1();
	for (int i = 0; i < w.size(); i++)
	{
		w[i] = rand1(); // random filling
	}
}

void Neuron::correct(double c)
{
	a = 0.0;
	er = 0.0;
	b -= db * c;
	db = 0.0;
	exp = 0.0;
	for (int i = 0; i < w.size(); i++)
	{
		w[i] -= dw[i] * c;
		dw[i] = 0.0;
	}
}

void Neuron::calcForward(vector<double>& prev)
{
	a = b;
	for (int i = 0; i < w.size(); i++)
	{
		a += prev[i] * w[i];
	}// gather all prev column
	a = sigma(a);
}

void Neuron::calcForward(Neurocolumn& prev)
{
	a = b;
	for (int i = 0; i < w.size(); i++)
	{
		a += prev.neu[i].a * w[i];
	}
	a = sigma(a);
}

///////////////////////////////////////////////////////////////////////////

void Neurocolumn::init(int n, int nPrev)
{
	neu.resize(n);
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].init(nPrev);
	}
}

void Neurocolumn::calcForward(vector<double>& prev)
{
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].calcForward(prev);
	}
}

void Neurocolumn::calcForward(Neurocolumn& prev)
{
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].calcForward(prev);
	}
}

void Neurocolumn::calcBack(Neurocolumn& prev)
{
	for (int i = 0; i < prev.neu.size(); i++)
	{
		prev.neu[i].exp = 0.0;
	}

	for (int i = 0; i < neu.size(); i++)
	{ //////////////// c is error = sqr(exp - a), a - activation, z = a * w + b - value inside sigma(z)
		double er = 2.0 * (neu[i].a - neu[i].exp) * dsigma(neu[i].a);//// er = dc/da * da/dz   here is hardcoded to logical sigma
		neu[i].db += er;
		for (int j = 0; j < neu[i].w.size(); j++)
		{
			neu[i].dw[j] += er * prev.neu[j].a; // dc/dw = dc/da * da/dz * dz/dw = er * a;
			prev.neu[j].exp += er * neu[i].w[j]; // dc/da_prev = dc/da * da/dz * dz/da_ = er * w ; a_ - a from prev
		}
	}
	for (int i = 0; i < prev.neu.size(); i++)
	{
		prev.neu[i].exp = prev.neu[i].a - prev.neu[i].exp; // calc expected values for previous column
	}
}

void Neurocolumn::calcBack(vector<double>& prev) // similar ar previous function but need not to calc a_
{
	for (int i = 0; i < neu.size(); i++)
	{
		double er = (neu[i].a - neu[i].exp) * dsigma(neu[i].a);///////////////// here is hardcoded to exp sigma
		neu[i].db += er;
		for (int j = 0; j < neu[i].w.size(); j++)
		{
			neu[i].dw[j] += er * prev[j];
		}
	}
}

void Neurocolumn::correct(double c)
{
	for (int i = 0; i < neu.size(); i++)
	{
		neu[i].correct(c);
	}
}

//////////////////////////////////////////////////////////////////////

void Neuronet::init(vector<int>& cols)
{
	column.resize(cols.size() - 1);
	for (int i = 1; i < cols.size(); i++)
	{
		column[i - 1].init(cols[i], cols[i - 1]); // we need n Neurons in this column an in prev (for weights)
	}
}

void Neuronet::calcForward(vector<double>& in) 
{
	column.front().calcForward(in);
	for (int i = 1; i < column.size(); i++)
	{
		column[i].calcForward(column[i - 1]);
	}
}

void Neuronet::calcBack(vector<double>& in)
{
	for (int i = column.size() - 1; i > 0; i--)
	{
		column[i].calcBack(column[i - 1]);
	}
	column.front().calcBack(in);
}

void Neuronet::setExp(vector<double>& out)
{
	error = 0.0;
	for (int i = 0; i < out.size(); i++)
	{
		column.back().neu[i].exp = out[i]; // put output 
		error += sqr(out[i] - column.back().neu[i].a); // and calc error in last column
	}
}

void Neuronet::correct(double c)
{
	for (int i = 0; i < column.size(); i++)
	{
		column[i].correct(c);
	}
}

void  Neuronet::calcForwardBack(vector<double>& in, vector<double>& out)
{
	calcForward(in); // forward
	setExp(out);   // get output
	calcBack(in);  // go back
	total_error += error; // and error
}

void Neuronet::mixData(vector<vector<double>>& in, vector<vector<double>>& out)
{
	for (int i = 0; i < in.size(); i++) // so it's so terrible... dont look there, please.
	{ // swap two random data
		int n1 = rand() % in.size();
		int n2 = rand() % in.size();
		auto t1 = in[n1];
		in[n1] = in[n2];
		in[n2] = t1;
		auto t2 = out[n1];
		out[n1] = out[n2];
		out[n2] = t2;
	}
}

void Neuronet::learn(vector<vector<double>>& in, vector<vector<double>>& out, int n)
{
	total_error = 0.0; // nulling total error
	mixData(in, out); // mix vectors
	int iter = 0;
	for (int i = 0; i < in.size(); i++) // learn it n times
	{
		calcForwardBack(in[i], out[i]);
		iter++;
		if (iter == n)
		{
			correct(learnRate / iter); // and correct weights (w) and offsets (b) as mean rate of all miniset
			iter = 0;
		}
	}
	if (iter > 0) // if n is not multiple of in.size() we have little tail
	{
		correct(learnRate / iter);
	}


}
