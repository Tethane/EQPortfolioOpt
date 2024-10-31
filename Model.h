#pragma once

#include "API.h"

#define F_RISK_FREE 0
#define F_MARKET 1
#define F_HIGH_MINUS_LOW 2
#define F_SMALL_MINUS_BIG 3

namespace EQ
{
	// Factor Modelling (ex: Rm-Rf, or HML, SMB, MOM, etc)
	class Factor {
	public:
		Eigen::VectorXf values;
		float expected;
		float sigma; // Variance of the factor
		int tag;
	public:
		virtual void calculate() = 0;
	};

	class RiskFreeFactor : public Factor {
	public:
		RiskFreeFactor(Eigen::MatrixXf _data)
		{
			this->values = _data.col(0);
		}

		void calculate() override
		{
			// Format: Bond Yield 

			this->values = fixedYieldToMaturityToHoldingReturn(this->values, 10);
			this->expected = aMean(this->values);
			this->sigma = variance(this->values);
		}
	};

	class MarketFactor : public Factor {
	private:
		Eigen::MatrixXf data;
	public:
		MarketFactor(Eigen::MatrixXf _data) : data(_data)
		{
			this->values = Eigen::VectorXf(_data.rows() - 1);
			this->tag = F_MARKET;
		};

		void calculate() override
		{
			// Format: Market - Bond Yield

			int numRows = this->data.rows() - 1;

			Eigen::VectorXf rm(numRows);
			Eigen::VectorXf rf(numRows);

			rf = fixedYieldToMaturityToHoldingReturn(this->data.col(1), 10);

			for (int i = 0; i < numRows; ++i)
			{
				rm(i) = (this->data(i, 0) - this->data(i + 1, 0)) / this->data(i + 1, 0);

				this->values(i) = rm(i) - rf(i);
			}

			std::cout << aMean(this->values) << std::endl;

			this->expected = aMean(this->values);
			this->sigma = variance(this->values);
		}
	};

	class HighMinusLowFactor : public Factor {
	private:
		Eigen::MatrixXf data;
	public:
		HighMinusLowFactor(const Eigen::MatrixXf& _data) : data(_data)
		{
			this->values = Eigen::VectorXf(_data.rows() - 1);
			this->tag = F_HIGH_MINUS_LOW;
		};

		void calculate() override
		{
			// Format: Growth - Value (i.e Low, High)

			int numRows = this->data.rows() - 1;
			Eigen::VectorXf rh(numRows);
			Eigen::VectorXf rl(numRows);

			for (int i = 0; i < numRows; ++i)
			{
				rl(i) = (this->data(i, 0) - this->data(i + 1, 0)) / this->data(i + 1, 0);
				rh(i) = (this->data(i, 1) - this->data(i + 1, 1)) / this->data(i + 1, 1);
				
				this->values(i) = rl(i) - rh(i);
			}

			this->expected = aMean(this->values);
			this->sigma = variance(this->values);
		}
	};

	class SmallMinusBigFactor : public Factor {
	private:
		Eigen::MatrixXf data;
	public:
		SmallMinusBigFactor(const Eigen::MatrixXf& _data) : data(_data)
		{
			this->values = Eigen::VectorXf(data.rows() - 1);
			this->tag = F_SMALL_MINUS_BIG;
		};

		void calculate() override
		{
			// Format: Growth - Value (i.e Low, High)

			int numRows = this->data.rows() - 1;
			Eigen::VectorXf rs(numRows);
			Eigen::VectorXf rb(numRows);

			for (int i = 0; i < numRows; ++i)
			{
				rb(i) = (this->data(i, 0) - this->data(i + 1, 0)) / this->data(i + 1, 0);
				rs(i) = (this->data(i, 1) - this->data(i + 1, 1)) / this->data(i + 1, 1);

				this->values(i) = rs(i) - rb(i);
			}

			this->expected = aMean(this->values);
			this->sigma = variance(this->values);
		}
	};

	// Return Generating Process Model
	class Model {
	public:
		float alpha;
		Eigen::VectorXf betas; // List of betas of the return for N factor model (index 0 is alpha)
		float sigma; // Sigma of epsilon (unsystematic risk)
		float riskfree; // Expected risk-free rate
		float expected; // Expected return of asset
		std::vector<std::shared_ptr<Factor>> factors;
		Eigen::VectorXi factorTags; // List of factor tags to be used in the model
	public:

		void calibrateSVD(const Eigen::VectorXf& r, const Eigen::VectorXf& rf)
		{
			// SVD

			Eigen::MatrixXf factors(r.size(), this->factors.size());

			for (int i = 0; i < this->factors.size(); ++i)
			{
				factors.col(i) = this->factors[i]->values;
			}

			Eigen::MatrixXf design(r.size(), this->factors.size() + 1);
			design << Eigen::MatrixXf::Ones(r.size(), 1), factors;

			Eigen::JacobiSVD<Eigen::MatrixXf> svd(design, Eigen::ComputeThinU | Eigen::ComputeThinV);
			Eigen::VectorXf coefficients = svd.solve(r - rf);

			this->alpha = coefficients(0);
			this->betas = coefficients.tail(this->factors.size());

			Eigen::VectorXf residuals = (r - rf) - (design * coefficients);

			this->sigma = variance(residuals);

			this->riskfree = std::max(0.0f, aMean(rf));

			this->expected = this->alpha + this->riskfree;

			for (int i = 0; i < this->factors.size(); ++i)
			{
				this->expected += this->betas[i] * this->factors[i]->expected;
			}
		}

		void calibrateQR(const Eigen::VectorXf& r, const Eigen::VectorXf& rf)
		{
			// QR

			Eigen::MatrixXf factors(r.size(), this->factors.size());

			for (int i = 0; i < this->factors.size(); ++i)
			{
				factors.col(i) = this->factors[i]->values;
			}

			Eigen::MatrixXf design(r.size(), this->factors.size() + 1);
			design << Eigen::MatrixXf::Ones(r.size(), 1), factors;

			
			Eigen::VectorXf coefficients = design.householderQr().solve(r - rf);

			this->alpha = coefficients(0);
			this->betas = coefficients.tail(this->factors.size());

			Eigen::VectorXf residuals = (r - rf) - (design * coefficients);

			this->sigma = variance(residuals);

			this->riskfree = std::max(0.0f, aMean(rf));

			this->expected = this->alpha + this->riskfree;

			for (int i = 0; i < this->factors.size(); ++i)
			{
				this->expected += this->betas[i] * this->factors[i]->expected;
			}
		}

		float getVariance()
		{
			float variance = this->sigma;

			for(int i = 0; i < this->factors.size(); ++i)
			{
				variance += this->factors[i]->sigma * this->betas(i) * this->betas(i);
			}

			return variance;
		}

		float getRisk()
		{
			float variance = this->getVariance();
			return std::sqrtf(variance);
		}

		float getExpected()
		{
			float expected = this->alpha + this->riskfree;

			for (int i = 0; i < this->factors.size(); ++i)
			{
				expected += this->betas[i] * this->factors[i]->expected;
			}

			return expected;
		}
	};
}