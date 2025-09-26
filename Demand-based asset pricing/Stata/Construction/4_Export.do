/* Export.do (STATA)
	Export monthly stock data.
	by Ralph Koijen & Motohiro Yogo */


clear all
set more off
set type double

**************************************

		**** ADJUST PATH ****

**************************************
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

/* Load CRSP Monthly Stock */

u permno date shrcd profit Gat divA_be daret LNbe LNme beta if date>=td(1jan1970) & inlist(shrcd,10,11) using "`path'\Output\StocksM", clear 
/*
!!!!!!! Variable Ispmi not included because we don't have S&P500-Index !!!!!!!
*/

/* Construct variables */

egen int row = group(date)
egen int col = group(permno)

/* Label variables */

format date %tdCCYY-NN-DD

/* Export to csv file */

sort date permno

export delim date row col daret LNme LNbe profit Gat divA_be beta using "`path'\Output\Export1", replace
/*
!!!!!!! Variable Ispmi not included because we don't have S&P500-Index !!!!!!!
*/

/* Load Fama-French factors */

u date mktrf smb hml rf if inrange(date,td(1jan2000),td(31dec2023)) using "`path'\Data\Fama French\Fama-French Factors\Fama-French Factors Monthly", clear

/* Label variables */

format date %tdCCYY-NN-DD

/* Export to csv file */

sort date

export delim date rf mktrf smb hml using "`path'\Output\Export2", replace
