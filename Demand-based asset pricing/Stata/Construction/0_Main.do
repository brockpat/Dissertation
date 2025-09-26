/*###########################################################
				
				****	ADJUST PATHS	****

/*** To run these DoFiles, the path has to be adjusted in EACH DoFile.
Unfortuantely, only locals can be used to
capture file paths adequately for our Stata version ***/

##############################################################*/

/* 1) Prepare CRSP & Compustat data. */
	/*
	Data Input: 
		CRSP_Compustat_Merged_Quarterly
			Source: WRDS.	Link: CRSP/Annual Update/ CRSP/Compustat Merged/ Fundamentals Quarterly
		CRSP_Compustat_Merged_Annual
			Source: WRDS.	Link: CRSP/Annual Update/ CRSP/Compustat Merged/ Fundamentals Annual
		CRSP_Monthly_Stock
			Source: WRDS.	SFTP Link: wrdslin/crsp/sasdata/a_stock/msf
							Link: CRSP/Annual Update/Stock/Security Files/Monthly Stock File
							
	Output: 
		Compustat.dta
		Stocks_Monthly.dta
	*/
do 1_Stocks_Monthly.do
**************************************************************
/* 2) Estimate rolling beta. */
//Caution: Can take a while
	/*
	Data Input: 
		Stocks_Monthly.dta
		Fama-French Factors Monthly
			Source: WRDS.	Link: Fama-French Portfolios and Factors/ Fama-French Portfolios /
									5 Factors Plus Momentum - Monthly Frequency
	
	Output: 
		Beta.dta
	*/
do 2_Beta
**************************************************************
/* 3) Construct monthly & quarterly stock data. */
	/*
	Data Input: 
		Stocks_Monthly.dta
		Beta.dta
			Source: WRDS.	Link: Fama-French Portfolios and Factors/ Fama-French Portfolios /
									5 Factors Plus Momentum - Monthly Frequency
	
	Output: 
		StocksM.dta
		StocksQ.dta
	*/
do 3_Stocks
*************************************************************
/* 4) Export monthly stock data. */
	/*
	Data Input: 
		StocksM.dta
		Fama-French Factors Monthly
	
	Output: 
		Export1.csv
		Export2.csv
	*/
do 4_Export
**************************************************************
/* 5) Construct SEC Form ADV data. */
//Caution: Can take a while
	/*
	Data Input: 
		See ReadMe in Data/SEC (too many input files to list).
		s34 Manager
			Source: WRDS.	SFTP Link: /wrdslin/tfn/sasdata/s34/s34type1.sas7bdat
							Link: Thomson/Refinitiv / Institutional (13f) Holdings - S34 / 
									Type 1:Manager
	
	Output: 
		SEC.dta
	*/
do 5_SEC
**************************************************************
/* 6) Construct Thomson Reuters manager data. */
	/*
	Data Input: 
		s12type5
			Source: WRDS.	SFTP Link: wrdslin/tfn/sasdata/s12/s12type5.sas7bdat
		Owner_Information
			Source: WRDS.	Thomson/Refinitiv / Global Ownership / Owner Information
		wrds_13f_link
			Source: WRDS.	SFTP Link: /wrds/sec/sasdata/wrds_13f_link
		s34 Manager (WARNING: we don't have official access to this file)
		SEC.dta
		Pension Funds_KY.dta
	
	Output: 
		UPDATED Manager.dta. KY19 provide their Manager.dta for free, but of course only until year 2017.
	*/
do 6_Manager
**************************************************************
/* 7) Fix Typecodes in SEC 13F data */
	/*
	Data Input: 
		Manager.dta from KY19 Replication package. KY19 provide this package to everyone, no license required
		SEC_13F from WRDS SEC Analytics Suite and Michael Sinkinson Scraping
			
	Output: 
		SEC_13F_Typecode
	*/
do 7_FixTypecode
**************************************************************
/* 8) Construct stock holdings data. Either from Thomson Reuters database (which we cannot officially) or from SEC data. */
//Caution: Can take a while
	/*
	Data Input: 
		13F_ThomsonReuters_1980-2023 
			Source: WRDS.	Link: Thomson/Refinitiv / Institutional (13f) Holdings - S34/
									s34 Master Files/Monthly
			Alternative Source: WRDS Thomson Reuters Global Owernship Institutional Holdings
								(Type 2)
			
		Manager.dta
		StocksQ.dta
		
	Output: 
		Holdings.dta
	*/
do 8_Holdings

