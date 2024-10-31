#pragma once

#include "API.h"
#include "Model.h"
#include "Asset.h"

namespace EQ
{
	// Format one row of comma separated names
	std::vector<std::string> getListFromString(const std::string& s)
	{
		std::vector<std::string> res;
		std::stringstream ss(s);
		std::string name;

		while (std::getline(ss, name, ','))
		{
			size_t start = name.find_first_not_of(" \t\n");
			size_t end = name.find_last_not_of(" \t\n");

			if (start != std::string::npos && end != std::string::npos)
			{
				res.push_back(name.substr(start, end - start + 1));
			}
			else
			{
				// This case handles if a name contains only whitespaces
				res.push_back("");
			}
		}

		return res;
	}

	// Search for specific stocks within database
	// - Since assets are already in alphabetical order, indices will automatically be in increasing order (thus we know that ith index associates to ith stock and reverse)
	std::vector<int> findColumnIndices(const std::string& filePath, const std::vector<std::string>& assets)
	{
		std::vector<int> indices;

		std::ifstream file(filePath);
		if (!file.is_open())
		{
			std::cerr << "Error opening file: " << filePath << std::endl;
			return indices;
		}

		std::string line;
		if (std::getline(file, line))
		{
			std::istringstream iss(line);
			std::string columnName;
			int columnIndex = 1;

			while (std::getline(iss, columnName, ','))
			{
				auto it = std::find(assets.begin(), assets.end(), columnName);
				if (it != assets.end())
				{
					indices.push_back(columnIndex - 2);
				}
				++columnIndex;
			}
		}

		file.close();

		return indices;
	}


	std::vector<std::pair<Asset, Model>> getAssetsFromListSameModel(const std::vector<std::string>& assets, const Model& model)
	{
		std::vector<std::pair<Asset, Model>> res;
		for (int i = 0; i < assets.size(); ++i)
		{
			Asset a;
			a.name = assets[i];

			if (a.name == "^SPX")
			{
				a.name = "S&P 500";
			}

			res.push_back(std::make_pair(a, model));
		}

		return res;
	}
}