#pragma once

#include <Eigen/Eigen>

#include <cmath>
#include <ctime>
#include <cstdlib>
#include <vector>
#include <iostream>
#include <string>
#include <fstream>
#include <tuple>
#include <memory>
#include <sstream>

#define ONE_MONTH 4
#define THREE_MONTH 13
#define SIX_MONTH 26
#define ONE_YEAR 52
#define THREE_YEAR 156
#define FIVE_YEAR 260
#define TEN_YEAR 520

namespace EQ
{
	// Arithmetic Mean (Expected Value)
	float aMean(const Eigen::VectorXf& data)
	{
		float sum = 0.0f;
		
		for (int i = 0; i < data.size(); ++i)
		{
			sum += data(i);
		}

		return sum / (float)(data.size());
	}

	// Geometric Mean (Expected Value)
	float gMean(const Eigen::VectorXf& data)
	{
		float logSum = 0.0f;

		for (int i = 0; i < data.size(); ++i)
		{
			std::cout << "Log value " << data(i) << " " << std::logf(data(i)) << std::endl;
			logSum += std::logf(data(i));
		}

		return std::expf(logSum / data.size());
	}

	// Variance
	float variance(const Eigen::VectorXf& data)
	{
		float mean = data.mean();
		float variance = (data.array() - mean).square().sum() / (data.size());
		
		return variance;
	}

	Eigen::VectorXf fixedYieldToMaturityToHoldingReturn(const Eigen::VectorXf& yields, int maturity)
	{
		float m = (float)maturity;

		Eigen::VectorXf res(yields.size() - 1);

		for (int i = 0; i < yields.size() - 1; ++i)
		{
			float y2 = yields[i]; // Yt
			float y1 = yields[i + 1]; // Yt-1
			float D = (1.0f / y2) * (1.0f - (1.0f / std::powf(1.0f + 0.5f * y2, 2.0f * m)));
			float C = (2.0f / (y2 * y2)) * (1.0f - (1.0f / std::powf(1.0f + 0.5f * y2, 2.0f * m))) - ((2.0f * m) / (y2 * std::powf(1.0f + 0.5f * y2, 2.0f * m + 1.0f)));
			float R = -D * (y2 - y1) + 0.5f * C * (y2 - y1) * (y2 - y1);

			res(i) = R / 100.0f;
		}

		return res;
	}

	int getIndexOfI(const std::vector<int>& data, int val)
	{
		for (int i = 0; i < data.size(); ++i) {
			if (data[i] == val) {
				return i;
			}
		}
		return -1; // Value not found
	}

	float annualReturnToWeekly(float annualReturn)
	{
		return std::powf((1.0f + annualReturn / 100.0f), (1.0f / 52.0f)) - 1.0f;
	}

	float erf_inv(float x) {
		const float PI = 3.14159265358979323846f;
		const float A0 = 2.50662823884f;
		const float A1 = -18.61500062529f;
		const float A2 = 41.39119773534f;
		const float A3 = -25.44106049637f;
		const float B1 = -8.47351093090f;
		const float B2 = 23.08336743743f;
		const float B3 = -21.06224101826f;
		const float B4 = 3.13082909833f;

		float t = 1.0f / (1.0f + 0.2316419f * std::abs(x));
		float y = 1.0f - (((((A3 * t + A2) * t) + A1) * t + A0) / (((((B4 * t + B3) * t + B2) * t + B1) * t + 1) * t));

		return x >= 0 ? y : -y;
	}

	float zScore(float percentage)
	{
		float prob = percentage / 100.0f;
		return std::sqrt(2.0f) * erf_inv(2.0f * prob - 1.0f);
	}
}