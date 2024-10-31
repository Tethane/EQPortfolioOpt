#pragma once

#include "API.h"
#include "Asset.h"
#include "Model.h"

namespace EQ
{
	class Portfolio {
	public:
		std::vector<std::tuple<Asset, Model>> investments;
	private:
		std::vector<std::shared_ptr<Factor>> factors; // List of all factors (each model will have a subset of these factors)
		Eigen::VectorXf benchmark; // Returns of the benchmark portfolio
		float benchmarkReturn; // Expected return of the benchmark portfolio (passive / market portfolio)
		float benchmarkRisk; // Variance of returns of the benchmark portfolio
		Eigen::VectorXf riskfree; // Historical risk free rates
		float riskfreeReturn; // Expected return of the risk free rate
		Eigen::VectorXf weights; // Weights for each of the individual assets (active), the last two elements of this vector represent the market weight and risk-free weight respectively
		std::vector<int> indices;
		float portfolioRisk;
		float portfolioExpectedReturn;
	public:
		Portfolio(const std::vector<std::pair<Asset, Model>>& assets)
		{
			for (int i = 0; i < assets.size(); ++i)
			{
				this->investments.push_back(std::make_tuple(assets[i].first, assets[i].second));
				this->indices.push_back(i);
			}
		}

		// Assets MUST BE IN ALPHABETICAL ORDER
		Portfolio(const std::vector<std::pair<Asset, Model>>& assets, const std::vector<int>& _indices) : indices(_indices)
		{
			for (int i = 0; i < indices.size(); ++i)
			{
				this->investments.push_back(std::make_tuple(assets[i].first, assets[i].second));
			}
		}

		void load(const std::string& benchmarkPath, const std::string& riskfreePath, const std::string& assetsPath, const std::vector<std::pair<int, std::string>>& factorsPaths, int T = 0)
		{
			// Process Factors
			// 1. For each of the factor data files provided, load data
			// 2. Process the data according to the tag of the factor

			for (int i = 0; i < factorsPaths.size(); ++i)
			{
				Eigen::MatrixXf factorData = this->loadData(factorsPaths[i].second, T);

				std::shared_ptr<Factor> factor = nullptr;

				switch (factorsPaths[i].first)
				{
				case F_RISK_FREE:
					factor = std::make_shared<RiskFreeFactor>(factorData);
					break;
				case F_MARKET:
					factor = std::make_shared<MarketFactor>(factorData);
					break;
				case F_HIGH_MINUS_LOW:
					factor = std::make_shared<HighMinusLowFactor>(factorData);
					break;
				case F_SMALL_MINUS_BIG:
					factor = std::make_shared<SmallMinusBigFactor>(factorData);
					break;
				default:
					break;
				}

				factor->calculate();

				this->factors.push_back(factor);
			}

			// Process Benchmark Returns
			
			Eigen::MatrixXf data = this->loadData(benchmarkPath, T);

			int numRows = data.rows() - 1;
			
			Eigen::VectorXf benchmarkReturns(numRows);

			for (int i = 0; i < numRows; ++i)
			{
				// benchmarkReturns(i) = (data(i, 0) - data(i + 1, 0)) / data(i + 1, 0);
				benchmarkReturns(i) = data(i, 0) - data(i + 1, 0);
			}

			this->benchmark = benchmarkReturns;
			this->benchmarkReturn = aMean(this->benchmark);
			this->benchmarkRisk = std::sqrtf(variance(this->benchmark));

			// Process Risk Free Rate

			data = this->loadData(riskfreePath, T);

			numRows = data.rows() - 1;
			int maturity = 10;

			Eigen::VectorXf riskfreeReturns(numRows);

			riskfreeReturns = fixedYieldToMaturityToHoldingReturn(data.col(0), maturity);

			this->riskfree = riskfreeReturns;
			this->riskfreeReturn = std::max(0.0f, aMean(this->riskfree));

			// Process Assets
			// 1. Load quotes

			data = this->loadData(assetsPath, T);

			// 2. Convert into holding returns
			// 3. Fill out the asset historical returns for each asset (default model is Fama-French 3 Factor Model)
			// - Smarter: Do this but only for specific assets as are chosen (select specific assets out of the library)

			numRows = data.rows() - 1;
			int numCols = data.cols();

			Eigen::MatrixXf returns(numRows, numCols);

			for (int i = 0; i < numCols; ++i)
			{
				if (std::find(this->indices.begin(), this->indices.end(), i) != this->indices.end())
				{
					for (int j = 0; j < numRows; ++j)
					{
						if (data(j + 1, i) != 0.0f)
						{
							// returns(j, i) = (data(j, i) - data(j + 1, i)) / data(j + 1, i);
							returns(j, i) = data(j, i) - data(j + 1, i);
						}
						else
						{
							if (data(j, i) == 0.0f)
							{
								returns(j, i) = 0.0f;
							}
							else
							{
								returns(j, i) = 1.0f;
							}
						}
					}

					int index = getIndexOfI(this->indices, i);
					std::get<0>(this->investments[index]).returns = returns.col(i);
					std::get<0>(this->investments[index]).mu = aMean(returns.col(i));

					for (int j = 0; j < std::get<1>(this->investments[index]).factorTags.size(); ++j)
					{
						int fac = std::get<1>(this->investments[index]).factorTags[j]; // Factor be included in the specific model

						for (int k = 0; k < this->factors.size(); ++k)
						{
							if (this->factors[k]->tag == fac)
							{
								std::get<1>(this->investments[index]).factors.push_back(this->factors[k]);
							}
						}
					}
				}
			}
		}

		Eigen::MatrixXf loadData(const std::string& filePath, int T = 0) const
		{
			std::ifstream file(filePath);

			if (!file.is_open())
			{
				std::cerr << "Error opening file: " << filePath << std::endl;
				return Eigen::MatrixXf();
			}

			std::vector<std::vector<float>> data;

			std::string header;
			std::getline(file, header);

			std::string line;
			while (std::getline(file, line))
			{
				std::vector<float> row;
				size_t pos = 0;

				while ((pos = line.find(',')) != std::string::npos)
				{
					std::string token = line.substr(0, pos);
					row.push_back(std::stof(token));
					line.erase(0, pos + 1);
				}

				row.push_back(std::stof(line));
				data.push_back(row);
			}

			file.close();

			int numRows = data.size();
			int numCols = data[0].size();
			Eigen::MatrixXf matrix(numRows, numCols - 1);
			for (int i = 0; i < numRows; ++i)
			{
				for (int j = 1; j < numCols; ++j)
				{
					matrix(i, j - 1) = std::logf(data[data.size() - 1 - i][j]); // Automatically sorts in DESCENDING ORDER (most recent at the top / first index)
				}
			}

			if (T == 0)
			{
				T = matrix.rows();
			}

			Eigen::MatrixXf res = matrix.topRows(T);

			return res;
		}

		// Min Variance for Target Return
		void optimizeMinVariance(float target)
		{
			// Initial Data
			// - Expected return of the market portfolio
			// - Variance of the market portfolio
			// - Expected risk free rate
			// - Target return of the optimal portfolio


			// 1. Estimate alpha, betas, sigma for each of the assets

			for (int i = 0; i < this->investments.size(); ++i)
			{
				Eigen::VectorXf r = std::get<0>(this->investments[i]).returns;
				Eigen::VectorXf rf = this->riskfree;

				std::get<1>(this->investments[i]).calibrateSVD(r, rf);
			}

			// 2. Compute the normalized weights in the active portfolio
			// - Calculate Lambda (solve the lagrangian)
			
			Eigen::VectorXf A(this->investments.size());
			float B = (this->benchmarkReturn - this->riskfreeReturn) / (this->benchmarkRisk);

			for (int i = 0; i < this->investments.size(); ++i)
			{
				float top = std::get<1>(this->investments[i]).alpha;
				float bottom = std::get<1>(this->investments[i]).sigma;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					top += std::get<1>(this->investments[i]).factors[j]->expected * std::get<1>(this->investments[i]).betas(j);

					float b = std::get<1>(this->investments[i]).betas(j);
					float s = std::get<1>(this->investments[i]).factors[j]->sigma;

					bottom += b * b * s;
				}

				A(i) = top / bottom;
			}

			float C = B * (this->benchmarkReturn - this->riskfreeReturn);

			for (int i = 0; i < this->investments.size(); ++i)
			{
				C += A(i) * std::get<1>(this->investments[i]).alpha;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					C += std::get<1>(this->investments[i]).factors[j]->expected * A(i) * std::get<1>(this->investments[i]).betas(j);
				}
			}

			float lambda = (target - this->riskfreeReturn) / C;
			
			// - Calculate Weights
			
			Eigen::VectorXf wi(this->investments.size());
			float wm = lambda * B;
			float wa = 0.0f;
			
			for (int i = 0; i < this->investments.size(); ++i)
			{
				wi(i) = lambda * A(i);
				wa += wi(i);
			}

			// - Calculate Normalized Weights

			Eigen::VectorXf wiNormalized = wi / wa;

			// 3. Compute Expected Return of the Active Portfolio
			
			float eRa = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				eRa += wiNormalized(i) * std::get<1>(this->investments[i]).expected;
			}

			// 4. Compute the Weights of the Tangency Portfolio

			float waNormalized = (wa) / (wa + wm);
			float wmNormalized = 1.0f - waNormalized;

			float eRq = waNormalized * eRa + wmNormalized * this->benchmarkReturn;

			// 5. Determine the Tangency Portfolio Weight

			float wq = (target - this->riskfreeReturn) / (eRq - this->riskfreeReturn);

			wm = wq * wmNormalized;
			wa = wq * waNormalized;

			float wf = 1.0f - wq;

			// 6. Determine the actual weights

			for (int i = 0; i < this->investments.size(); ++i)
			{
				wi(i) = wa * wiNormalized(i);
			}
			
			// Normalize the weights one final time to cover floating point / division errors


			Eigen::VectorXf w(this->investments.size() + 2);
			w << wi, wm, wf;

			this->weights = w;

			float normalizationFactor = (1.0f / this->weights.sum());

			this->weights *= normalizationFactor;
		}

		void optimizeMaxReturn(float target)
		{
			// Initial Data
			// - Expected return of the market portfolio
			// - Variance of the market portfolio
			// - Expected risk free rate
			// - Target return of the optimal portfolio


			// 1. Estimate alpha, betas, sigma for each of the assets

			for (int i = 0; i < this->investments.size(); ++i)
			{
				Eigen::VectorXf r = std::get<0>(this->investments[i]).returns;
				Eigen::VectorXf rf = this->riskfree;

				std::get<1>(this->investments[i]).calibrateSVD(r, rf);
			}

			// 2. Compute the normalized weights in the active portfolio
			// - Calculate Lambda (solve the lagrangian)

			Eigen::VectorXf A(this->investments.size());
			float B = (this->benchmarkReturn - this->riskfreeReturn) / (2.0f * this->benchmarkRisk);

			for (int i = 0; i < this->investments.size(); ++i)
			{
				float top = std::get<1>(this->investments[i]).alpha;
				float bottom = std::get<1>(this->investments[i]).sigma;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					top += std::get<1>(this->investments[i]).factors[j]->expected * std::get<1>(this->investments[i]).betas(j);

					float b = std::get<1>(this->investments[i]).betas(j);
					float s = std::get<1>(this->investments[i]).factors[j]->sigma;

					bottom += b * b * s;
				}

				A(i) = (top) / (2.0f * bottom);
			}

			float C = B * B * this->benchmarkRisk;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				C += A(i) * A(i) * std::get<1>(this->investments[i]).sigma;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					float b = std::get<1>(this->investments[i]).betas(j);
					
					C += std::get<1>(this->investments[i]).factors[j]->expected * A(i) * A(i) * b * b;
				}
			}

			float lambda = std::sqrtf(C / target);

			std::cout << lambda << std::endl;

			// - Calculate Weights

			Eigen::VectorXf wi(this->investments.size());
			float wm = B / lambda;
			float wa = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				wi(i) = A(i) / lambda;
				wa += wi(i);
			}

			std::cout << wm << std::endl;
			std::cout << wa << std::endl;
			std::cout << "Hi" << std::endl;
			std::cout << wi << std::endl;


			// - Calculate Normalized Weights

			Eigen::VectorXf wiNormalized = wi / wa;

			// 3. Compute Variance of the Active Portfolio

			float eVa = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				eVa += wiNormalized(i) * wiNormalized(i) * std::get<1>(this->investments[i]).sigma;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					float b = std::get<1>(this->investments[i]).betas(j);

					eVa += wiNormalized(i) * wiNormalized(i) * b * b * std::get<1>(this->investments[i]).factors[j]->sigma;
				}
			}

			std::cout << eVa << std::endl;

			// 4. Compute the Weights of the Tangency Portfolio

			float waNormalized = wa / (wa + wm);
			float wmNormalized = 1.0f - waNormalized;

			float eVq = waNormalized * eVa + wmNormalized * (wm * wm * this->benchmarkRisk);
			// float eVq = wiNormalized.array().square().sum() * eVa + 

			// 5. Determine the Tangency Portfolio Weight

			float wq = target / eVq;

			std::cout << wq << std::endl;
			std::cout << eVq << std::endl;

			wm = wq * wmNormalized;
			wa = wq * waNormalized;

			float wf = 1.0f - wq;

			// 6. Determine the actual weights

			for (int i = 0; i < this->investments.size(); ++i)
			{
				wi(i) = wa * wiNormalized(i);
			}

			// Normalize the weights one final time to cover floating point / division errors


			Eigen::VectorXf w(this->investments.size() + 2);
			w << wi, wm, wf;

			this->weights = w;

			float normalizationFactor = (1.0f / this->weights.sum());

			this->weights *= normalizationFactor;
		}
		
		void optimizeMaxReturn1(float target)
		{
			// Initial Data
			// - Expected return of the market portfolio
			// - Variance of the market portfolio
			// - Expected risk free rate
			// - Target return of the optimal portfolio


			// 1. Estimate alpha, betas, sigma for each of the assets

			for (int i = 0; i < this->investments.size(); ++i)
			{
				Eigen::VectorXf r = std::get<0>(this->investments[i]).returns;
				Eigen::VectorXf rf = this->riskfree;

				std::get<1>(this->investments[i]).calibrateSVD(r, rf);
			}

			// 2. Compute the normalized weights in the active portfolio
			// - Calculate Lambda (solve the lagrangian)

			Eigen::VectorXf A(this->investments.size());
			float B = (this->benchmarkReturn - this->riskfreeReturn) / (2.0f * this->benchmarkRisk);

			for (int i = 0; i < this->investments.size(); ++i)
			{
				float top = std::get<1>(this->investments[i]).alpha;
				float bottom = std::get<1>(this->investments[i]).sigma;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					top += std::get<1>(this->investments[i]).factors[j]->expected * std::get<1>(this->investments[i]).betas(j);

					float b = std::get<1>(this->investments[i]).betas(j);
					float s = std::get<1>(this->investments[i]).factors[j]->sigma;

					bottom += b * b * s;
				}

				A(i) = (top) / (2.0f * bottom);
			}

			float C = B * B * this->benchmarkRisk;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				C += A(i) * A(i) * std::get<1>(this->investments[i]).sigma;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					float b = std::get<1>(this->investments[i]).betas(j);

					C += std::get<1>(this->investments[i]).factors[j]->expected * A(i) * A(i) * b * b;
				}
			}

			float lambda = std::sqrtf(C / target);

			std::cout << lambda << std::endl;

			// - Calculate Weights

			Eigen::VectorXf wi(this->investments.size());
			float wm = B / lambda;
			float wa = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				wi(i) = A(i) / lambda;
				wa += wi(i);
			}

			


			std::cout << wm << std::endl;
			std::cout << wa << std::endl;
			std::cout << "Hi" << std::endl;
			std::cout << wi << std::endl;


			// - Calculate Normalized Weights

			Eigen::VectorXf wiNormalized = wi / wa;

			// 3. Compute Variance of the Active Portfolio

			float eVa = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				eVa += wiNormalized(i) * wiNormalized(i) * std::get<1>(this->investments[i]).sigma;

				for (int j = 0; j < std::get<1>(this->investments[i]).factors.size(); ++j)
				{
					float b = std::get<1>(this->investments[i]).betas(j);

					eVa += wiNormalized(i) * wiNormalized(i) * b * b * std::get<1>(this->investments[i]).factors[j]->sigma;
				}
			}

			std::cout << eVa << std::endl;

			// 4. Compute the Weights of the Tangency Portfolio

			float waNormalized = wi.array().square().sum() / (wi.array().square().sum() + wm * wm);
			float wmNormalized = 1.0f - waNormalized;

			float eVq = waNormalized * eVa + wmNormalized * (wm * wm * this->benchmarkRisk);
			// float eVq = wiNormalized.array().square().sum() * eVa + 

			// 5. Determine the Tangency Portfolio Weight

			float wq = (target / eVq);

			std::cout << wq << std::endl;
			std::cout << eVq << std::endl;

			wm = wq * wmNormalized;
			wa = wq * waNormalized;

			float wf = 1.0f - wq;

			// 6. Determine the actual weights

			for (int i = 0; i < this->investments.size(); ++i)
			{
				wi(i) = wa * wiNormalized(i);
			}

			// Normalize the weights one final time to cover floating point / division errors


			Eigen::VectorXf w(this->investments.size() + 2);
			w << wi, wm, wf;

			this->weights = w;

			float normalizationFactor = (1.0f / this->weights.sum());

			this->weights *= normalizationFactor;
		}

		void optimizeTargetRisk(float target)
		{
			// 1. Estimate alpha, betas, sigma for each of the assets

			for (int i = 0; i < this->investments.size(); ++i)
			{
				Eigen::VectorXf r = std::get<0>(this->investments[i]).returns;
				Eigen::VectorXf rf = this->riskfree;

				std::get<1>(this->investments[i]).calibrateSVD(r, rf);
			}

			// Get the alphas
			Eigen::VectorXf alphas(this->investments.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				alphas(i) = std::get<1>(this->investments[i]).alpha;
			}

			// Get the betas
			Eigen::MatrixXf betas(this->investments.size(), std::get<1>(this->investments[0]).betas.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				for (int j = 0; j < std::get<1>(this->investments[i]).betas.size(); ++j)
				{
					betas(i, j) = std::get<1>(this->investments[i]).betas(j);
				}
			}

			// Get the residuals
			Eigen::VectorXf residuals(this->investments.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				residuals(i) = std::get<1>(this->investments[i]).sigma;
			}

			// Get the covariance matrix

			Eigen::MatrixXf var = Eigen::MatrixXf::Zero(this->investments.size(), this->investments.size());

			for (int i = 0; i < std::get<1>(this->investments[0]).factors.size(); ++i)
			{
				var += std::get<1>(this->investments[i]).factors[i]->sigma * betas.col(i) * betas.col(i).transpose();
			}

			Eigen::MatrixXf D = residuals.asDiagonal();

			var += D;

			// Get the expected returns vector

			Eigen::VectorXf expected(this->investments.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				expected(i) = alphas(i);

				for(int j = 0; j < std::get<1>(this->investments[0]).factors.size(); ++j)
				{
					expected(i) += betas(i, j) * std::get<1>(this->investments[i]).factors[j]->expected;
				}
			}

			// Compute the Tangency Portfolio Weights

			Eigen::MatrixXf invVar = var.inverse();
			Eigen::VectorXf ones = Eigen::VectorXf::Ones(this->investments.size());

			Eigen::VectorXf t = (invVar * (expected - this->riskfreeReturn * ones)) / (ones.transpose() * invVar * (expected - this->riskfreeReturn * ones));

			// Solve for the tangency portfolio weight

			float tExpected = t.transpose() * expected;
			float tVar = t.transpose() * var * t;
			float tSD = std::sqrtf(tVar);

			float xt = target / tSD;
			float xf = 1.0f - xt;

			// Solve for the efficient portfolio weights
			Eigen::VectorXf xti = t * xt;

			Eigen::VectorXf w(this->investments.size() + 1);
			w << xti, xf;

			this->weights = w;

			this->portfolioRisk = target;
			this->portfolioExpectedReturn = xt * tExpected + xf * this->riskfreeReturn;
		}

		void optimizeTargetReturn(float target)
		{
			// 1. Estimate alpha, betas, sigma for each of the assets

			for (int i = 0; i < this->investments.size(); ++i)
			{
				Eigen::VectorXf r = std::get<0>(this->investments[i]).returns;
				Eigen::VectorXf rf = this->riskfree;

				std::get<1>(this->investments[i]).calibrateSVD(r, rf);
			}

			// Get the alphas
			Eigen::VectorXf alphas(this->investments.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				alphas(i) = std::get<1>(this->investments[i]).alpha;
			}

			// Get the betas
			Eigen::MatrixXf betas(this->investments.size(), std::get<1>(this->investments[0]).betas.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				for (int j = 0; j < std::get<1>(this->investments[i]).betas.size(); ++j)
				{
					betas(i, j) = std::get<1>(this->investments[i]).betas(j);
				}
			}

			// Get the residuals
			Eigen::VectorXf residuals(this->investments.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				residuals(i) = std::get<1>(this->investments[i]).sigma;
			}

			// Get the covariance matrix

			Eigen::MatrixXf var = Eigen::MatrixXf::Zero(this->investments.size(), this->investments.size());

			for (int i = 0; i < std::get<1>(this->investments[0]).factors.size(); ++i)
			{
				var += std::get<1>(this->investments[i]).factors[i]->sigma * betas.col(i) * betas.col(i).transpose();
			}

			Eigen::MatrixXf D = residuals.asDiagonal();

			var += D;

			// Get the expected returns vector

			Eigen::VectorXf expected(this->investments.size());

			for (int i = 0; i < this->investments.size(); ++i)
			{
				expected(i) = alphas(i);

				for (int j = 0; j < std::get<1>(this->investments[0]).factors.size(); ++j)
				{
					expected(i) += betas(i, j) * std::get<1>(this->investments[i]).factors[j]->expected;
				}
			}

			// Compute the Tangency Portfolio Weights

			Eigen::MatrixXf invVar = var.inverse();
			Eigen::VectorXf ones = Eigen::VectorXf::Ones(this->investments.size());

			Eigen::VectorXf t = (invVar * (expected - this->riskfreeReturn * ones)) / (ones.transpose() * invVar * (expected - this->riskfreeReturn * ones));

			// Solve for the tangency portfolio weight

			float tExpected = t.transpose() * expected;
			float tVar = t.transpose() * var * t;
			float tSD = std::sqrtf(tVar);

			float xt = (target - this->riskfreeReturn) / (tExpected - this->riskfreeReturn);
			float xf = 1.0f - xt;

			// Solve for the efficient portfolio weights
			Eigen::VectorXf xti = t * xt;

			Eigen::VectorXf w(this->investments.size() + 1);
			w << xti, xf;

			this->weights = w;

			this->portfolioRisk = xt * tSD;
			this->portfolioExpectedReturn = xt * tExpected + xf * this->riskfreeReturn;
		}

		float getPortfolioRisk() const
		{
			return this->portfolioRisk;
		}

		float getPortfolioExpectedReturn() const
		{
			return this->portfolioExpectedReturn;
		}

		float getMaxNonLeveragedReturn()
		{
			// From within the invested assets find the one with the highest expected return
			float greatestReturn = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				greatestReturn = std::fmaxf(greatestReturn, std::get<1>(this->investments[i]).expected);
			}

			return greatestReturn;
		}

		void showWeightsAndAllocatedFunds(float wealth)
		{
			std::cout << "As an investor with: $" << wealth << " you would want to invest accordingly:" << std::endl;

			for (int i = 0; i < this->weights.size() - 1; ++i)
			{
				std::cout << "=======================" << std::endl;
				std::cout << "Asset " << i + 1 << " : " << std::get<0>(this->investments[i]).name << std::endl;
				std::cout << "Expected Return: " << std::get<1>(this->investments[i]).getExpected() << std::endl;
				std::cout << "Risk: " << std::get<1>(this->investments[i]).getRisk() << std::endl;
				std::cout << "Weight: " << this->weights[i] << std::endl;
				std::cout << std::endl;
				std::cout << "Invest: $" << this->weights[i] * wealth << std::endl;
			}

			float portfolioExpectedReturn = this->getPortfolioExpectedReturn();
			float portfolioRisk = this->getPortfolioRisk();
			float portfolioVaR = this->calculateVaR(portfolioExpectedReturn, portfolioRisk, 99.0f) * wealth;
			//float portfolioBenefitRatio = this->calculateBenefitRatio(portfolioExpectedReturn, portfolioRisk, wealth, 99.0f);

			float benchmarkVaR = this->calculateVaR(this->benchmarkReturn, this->benchmarkRisk, 99.0f) * wealth;
			//float benchmarkBenefitRatio = this->calculateBenefitRatio(this->benchmarkReturn, this->benchmarkRisk, wealth, 99.0f);

			std::cout << "=======================" << std::endl;
			std::cout << "Weight for risk-free asset is: " << this->weights[this->weights.size() - 1] << std::endl;
			std::cout << "Risk-Free Portolio Investment: $" << this->weights[this->weights.size() - 1] * wealth << std::endl;
			std::cout << "=======================" << std::endl;
			std::cout << "Sum of weights: " << this->weights.sum() << std::endl;
			std::cout << "Maximum Un-Leveraged Return: " << this->getMaxNonLeveragedReturn() << std::endl;
			std::cout << "=======================" << std::endl;
			std::cout << "Expected Return of the Portfolio: " << portfolioExpectedReturn << std::endl;
			std::cout << "Risk of the Portfolio: " << portfolioRisk << std::endl;
			//std::cout << "Value at Risk (95%): " << this->calculateVaR(portfolioExpectedReturn, portfolioVariance, 95.0f) * wealth << std::endl;
			//std::cout << "Risk to Reward (95%): " << this->calculateBenefitRatio(portfolioExpectedReturn, portfolioVariance, wealth, 95.0f) << std::endl;
			std::cout << "Value at Risk (99%): -$" << portfolioVaR << std::endl;
			//std::cout << "Benefit Ratio (99%): " << portfolioBenefitRatio << std::endl;
			std::cout << "=======================" << std::endl;
			std::cout << "Expected Return of the Market: " << this->benchmarkReturn << std::endl;
			std::cout << "Risk of the Market: " << this->benchmarkRisk << std::endl;
			//std::cout << "Value at Risk (95%): " << this->calculateVaR(this->benchmarkReturn, this->benchmarkRisk, 95.0f) * wealth << std::endl;
			//std::cout << "Risk to Reward (95%): " << this->calculateBenefitRatio(this->benchmarkReturn, this->benchmarkRisk, wealth, 95.0f) << std::endl;
			std::cout << "Value at Risk (99%): -$" << benchmarkVaR << std::endl;
			//std::cout << "Benefit Ratio (99%): " << benchmarkBenefitRatio << std::endl;
			std::cout << "=======================" << std::endl;
		}

		float calculateExpectedReturn()
		{
			float ans = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				ans += this->weights[i] * std::get<1>(this->investments[i]).getExpected();
			}

			ans += this->riskfreeReturn * this->weights[this->weights.size() - 1];

			return ans;
		}

		float calculateVariance()
		{
			float ans = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				ans += this->weights[i] * this->weights[i] * std::get<1>(this->investments[i]).getVariance();
			}

			return ans;
		}

		float calculateRisk()
		{
			float ans = 0.0f;

			for (int i = 0; i < this->investments.size(); ++i)
			{
				ans += this->weights[i] * std::get<1>(this->investments[i]).getRisk();
			}

			return ans;
		}

		float calculateBenefitRatio(float expected, float sigma, float wealth, float confidence)
		{
			float VaR = this->calculateVaR(expected, sigma, confidence);

			float risk = wealth * VaR;

			float reward = (1.0f + expected) * wealth;

			float benefit = reward / std::abs(risk);

			return benefit;
		}

		float calculateVaR(float expected, float sigma, float confidence)
		{
			float stdd = sigma;
			float z = zScore(confidence);

			float VaR = expected - stdd * z;

			return std::abs(VaR);
		}
	};
}