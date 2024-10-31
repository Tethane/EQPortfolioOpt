#pragma once

#include "API.h"
#include "Model.h"

namespace EQ
{
	std::vector<std::pair<int, std::string>> getFactors()
	{
		auto riskFreeFactor = std::make_pair(F_RISK_FREE, "C:\\Users\\Owner\\Documents\\C#\\Data\\Factors\\DGS10.csv");
		auto marketFactor = std::make_pair(F_MARKET, "C:\\Users\\Owner\\Documents\\C#\\Data\\Factors\\Market.csv");
		auto highMinusLowFactor = std::make_pair(F_HIGH_MINUS_LOW, "C:\\Users\\Owner\\Documents\\C#\\Data\\Factors\\HighMinusLow.csv");
		auto smallMinusBigFactor = std::make_pair(F_SMALL_MINUS_BIG, "C:\\Users\\Owner\\Documents\\C#\\Data\\Factors\\SmallMinusBig.csv");

		std::vector<std::pair<int, std::string>> res;
		
		res.push_back(marketFactor);
		res.push_back(highMinusLowFactor);
		res.push_back(smallMinusBigFactor);

		return res;
	}
}