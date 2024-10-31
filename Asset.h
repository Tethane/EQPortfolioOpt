#pragma once

#include "API.h"

namespace EQ
{
	// Standard Investable Asset
	class Asset {
	public:
		Eigen::VectorXf returns;
		float mu;
		float sigma;
		std::string name; // Ticker
	};
}