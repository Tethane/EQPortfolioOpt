#include "API.h"
#include "Portfolio.h"
#include "AssetsList.h"
#include "FactorsList.h"

int main(int argc, char* argv[])
{
	// Define Factors List
	
	std::vector<std::pair<int, std::string>> factorsPaths = EQ::getFactors();

	// Define Assets and Models List

	std::string list = "AAL, AAPL, ABNB, ADBE, ADM, AG, ALB, AMAT, AMZN, ANF, AVGO, AWK, BAC, BKNG, BKR, BLK, BRK-B, BSX, CAT, CLDX, COIN, COST, DE, DIS, DOW, ELV, F, FCX, FDX, GCT, GD, GE, GIS, GS, H, HD, HSY, HUM, INTC, JNJ, JPM, KO, LMT, LSCC, LULU, MA, MCD, META, MMM, MRK, MRNA, MSFT, MU, NEE, NEM, NEO, NFLX, NKE, NVDA, PFE, PG, PLTR, POOL, PWR, PYPL, RACE, RTX, SEDG, SHOP, SPY, SYM, SYY, T, TGT, TMUS, TNA, TQQQ, TSLA, TXN, UNH, V, VST, VZ, WM, WMT, XEL, XOM, ^SPX";
	std::string list2 = "AAPL, AMZN, BAC, BRK-B, NVDA";

	std::vector<std::string> assets = EQ::getListFromString(list2);
	

	EQ::Model ffm;
	ffm.factorTags = Eigen::VectorXi(3);
	ffm.factorTags << F_MARKET, F_HIGH_MINUS_LOW, F_SMALL_MINUS_BIG;

	std::vector<std::pair<EQ::Asset, EQ::Model>> assetsList = EQ::getAssetsFromListSameModel(assets, ffm);

	// Create Portfolio Object

	std::string assetsPath = "C:\\Users\\Owner\\Documents\\C#\\Data\\Assets\\Stocks.csv";
	std::string assetsPath2 = "C:\\Users\\Owner\\Documents\\C#\\Data\\Assets\\Stocks2.csv";
	std::string benchmarkPath = "C:\\Users\\Owner\\Documents\\C#\\Data\\Assets\\Benchmark.csv";
	std::string riskfreePath = "C:\\Users\\Owner\\Documents\\C#\\Data\\Assets\\RiskFree.csv";

	std::vector<int> indices = EQ::findColumnIndices(assetsPath, assets);

	EQ::Portfolio portfolio(assetsList, indices);

	portfolio.load(benchmarkPath, riskfreePath, assetsPath, factorsPaths, ONE_YEAR);

	// portfolio.optimizeMinVariance(EQ::annualReturnToWeekly(165.0f));

	portfolio.optimizeTargetRisk(0.02);
	// portfolio.optimizeTargetReturn(EQ::annualReturnToWeekly(15.0f));

	portfolio.showWeightsAndAllocatedFunds(100000.0f);

	// portfolio.getMaxNonLeveragedReturn();

	return 0;
}