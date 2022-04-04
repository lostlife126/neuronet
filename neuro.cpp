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

Neurocolumn::Neurocolumn(int n, int nPrev):
	size_(n),
	sizePrev_(nPrev)
{
	a_.resize(n);
	b_.resize(n);
	db_.resize(n);
	exp_.resize(n);
	w_.resize(n);
	dw_.resize(n);
	for (int i = 0; i < n; i++)
	{
		a_[i] = 0.0;
		b_[i] = rand1();
		db_[i] = 0.0;
		exp_[i] = 0.0;
		w_[i].resize(nPrev);
		dw_[i].resize(nPrev);
		for (int j = 0; j < nPrev; j++)
		{
			w_[i][j] = rand1();
			dw_[i][j] = 0.0;
		}
	}
}

void Neurocolumn::calcActivation(vect& prev)
{
	for (int i = 0; i < size_; i++)
	{
		a_[i] = b_[i];
		for (int j = 0; j < sizePrev_; j++)
		{
			a_[i] += w_[i][j] * prev[j];
		}
	}
	for (int i = 0; i < size_; i++)
	{
		a_[i] = sigma(a_[i]);
	}
}

void Neurocolumn::calcActivation(Neurocolumn* prev)
{
	for (int i = 0; i < size_; i++)
	{
		a_[i] = b_[i];
		for (int j = 0; j < sizePrev_; j++)
		{
			a_[i] += w_[i][j] * prev->a_[j];
		}
	}
	for (int i = 0; i < size_; i++)
	{
		a_[i] = sigma(a_[i]);
	}
}

void Neurocolumn::calcBack(Neurocolumn* prev)
{

	for (int i = 0; i < sizePrev_; i++)
	{
		prev->exp_[i] = 0.0;
	}

	for (int i = 0; i < size_; i++)
	{ //////////////// c is error = sqr(exp - a), a - activation, z = a * w + b - value inside sigma(z)
		double error = 2.0 * (a_[i] - exp_[i]) * dsigma(a_[i]);//// er = dc/da * da/dz   here is hardcoded to logical sigma
		db_[i] += error;
		for (int j = 0; j < sizePrev_; j++)
		{
			dw_[i][j] += error * prev->a_[j]; // dc/dw = dc/da * da/dz * dz/dw = er * a;
			prev->exp_[j] += error * w_[i][j]; // dc/da_prev = dc/da * da/dz * dz/da_ = er * w ; a_ - a from prev
		}
	}
	for (int i = 0; i < sizePrev_; i++)
	{
		prev->exp_[i] = prev->a_[i] - prev->exp_[i];
	}

}

void Neurocolumn::calcBack(vect& prev) // similar ar previous function but need not to calc a_
{
	for (int i = 0; i < size_; i++)
	{ //////////////// c is error = sqr(exp - a), a - activation, z = a * w + b - value inside sigma(z)
		double error = 2.0 * (a_[i] - exp_[i]) * dsigma(a_[i]);//// er = dc/da * da/dz   here is hardcoded to logical sigma
		db_[i] += error;
		for (int j = 0; j < sizePrev_; j++)
		{
			dw_[i][j] += error * prev[j]; // dc/dw = dc/da * da/dz * dz/dw = er * a;
		}
	}
}

void Neurocolumn::correctAll(double c)
{
	for (int i = 0; i < size_; i++)
	{
		exp_[i] = 0.0;
		a_[i] = 0.0;
		b_[i] -= db_[i] * c;
		db_[i] = 0.0;
		for (int j = 0; j < sizePrev_; j++)
		{
			w_[i][j] -= dw_[i][j] * c;
			dw_[i][j] = 0.0;
		}
	}
}

void Neurocolumn::setExpected(vect& expected)
{
	for (int i = 0; i < size_; i++)
	{
		exp_[i] = expected[i];
	}
}

double Neurocolumn::calcError(vect& out)
{
	double error = 0.0;
	for (int i = 0; i < size_; i++)
	{
		error += sqr(out[i] - a_[i]);
	}
	return error;
}

//////////////////////////////////////////////////////////////////////

void Neuronet::init(vector<int>& cols)
{
	column.resize(cols.size() - 1);
	for (int i = 1; i < cols.size(); i++)
	{ // we need n Neurons in this column an in prev (for weights)
		column[i - 1] = new Neurocolumn(cols[i], cols[i - 1]);
	}
}

void Neuronet::calcForward(vect& in)
{
	column.front()->calcActivation(in);
	for (int i = 1; i < column.size(); i++)
	{
		column[i]->calcActivation(column[i - 1]);
	}
}

void Neuronet::calcBack(vect& in)
{
	for (int i = column.size() - 1; i > 0; i--)
	{
		column[i]->calcBack(column[i - 1]);
	}
	column.front()->calcBack(in);
}

void Neuronet::setExpected(vect& out)
{
	column.back()->setExpected(out);
	error = column.back()->calcError(out);
}

void Neuronet::correctAll(double c)
{
	for (auto& it : column)
	{
		it->correctAll(c);
	}
}

void  Neuronet::calcForwardBack(vector<double>& in, vector<double>& out)
{
	calcForward(in); // forward
	setExpected(out);   // get output
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
			correctAll((learnRate * iter) / n); // and correct weights (w) and offsets (b) as mean rate of all miniset
			iter = 0;
		}
	}
	if (iter > 0) // if n is not multiple of in.size() we have little tail
	{
		correctAll((learnRate * iter) / n);
	}


}