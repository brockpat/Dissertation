/*
******************		File Description		******************
						----------------
				
							PART I
				
This File merges SEC 13F Portfolio Holdings Data with stock characteristics & institution type codes. 
In doing so, it replaces the proprietary Thomson Reuters 13F Portfolio Data (s34) with the publically
available 13F Data from the SEC.
	
The 13F files are from 1999 to 2012 and scraped by Michael Sinkinson.
In contrast to the proprietary TR Data, the distinction between the report date (rdate)
and the file date (fdate) is important here. The report date is the relevant merging date as 
this is the date for which the portfolio data is being reported. One can clearly see 
that the fdate is mostly 45 days after the report date, although sometimes the file date is even much later.

Relevant Files are 
	1) SEC 13F data (SEC_Sinkinson_1999-2012)
		---> Identifier: CIK. Contains: Portfolio Holdings data of stocks identified by CUSIP.
	
	2) StocksQ_clean from KY19
		---> Identifier: CUSIP. Contains: Stock Characteristics
	
	3) Thomson Reuters Global Ownership Owner Information
		---> Identifier: CIK. Contains: raw Owner Type Code
		
	4) Manager.dta 
		---> Identifier: mgrno. Contains: Cleaned Owner Type Code
	5) WRDS 13F Link File
		----> Links CIK Identifier to mgrno Identifier
	
How to get the relevant files:
	The File SEC_Sinkinson_1999-2012 was modified in the sense that the raw date
	fits the quarterly date structure of the other datasets. In the raw data, the full
	date can be found (i.e. dd/mm/yyyy). Moreover, the cusip was truncated to 8 digits as the StocksQ
	File only contains the 8 digit cusip. 

	Thomson Reuters Global Ownership Owner Information was downloaded from WRDS (requires a subscription)
	
	WRDS 13F Link File was downloaded from WRDS (is available for free)
	
	StocksQ_clean is obtained from KY19's replication package (required CRSP/Compustat subscritpion).
	The file is cleaned in the sense that date formats and date variable names were edited for merging.
	
	Manager.dta is produced by the replication package of KY19 WHILE using the original s34 Thomson Reuters 13F data.
	
	It is possible to obtain the cleaned Owner Type Codes without Manager.dta
	by running KY19's Manager.do. 

Merging the Stock characteristics:
	The Stock characteristics can simply be merged through the cusip.
	
Merging the Owner Type Code:
	1) If Manager.dta is available:
		Complement CIK identifier with mgrno identifier via WRDS 13F Link File
		Merge Manager.dta via mgrno to datasets
	
	2) If Manager.dta NOT availalbe:
		Merge raw Thomson Reuters Owner Type Code via CIK from Thomson Reuters Global Ownership Owner Information.
		Execute the steps in Manager.do from KY19's replication package to clean the raw Owner Type Code. 
		
	
Important. The Stocks characteristics data must be lagged: 
	The Stock characteristics data (StocksQ) should be lagged for the following reason:
	Portfolio holdings are only reported at the end of the Quarter, but it is 
	likely that most trades happened at the beginning and middle of the quarter.
	Thus, at report date (rdate) t the institution most likely only had access
	to the stock characteristic data at time t-1 on which it based its holding decision.
	
 */

clear all
set more off
local path = "\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

/*

/**** Prepare Thomson Reuters Owner Type Code from Owner Information for merging ****/
u "`path'\Output\Owner_info"
drop if CIKCode == ""
*Drop duplicates, but keep first observation of duplicates
sort CIKCode
forval i = 1/4 {
	duplicates tag CIKCode, gen(dup)
	replace dup = 1 if dup > 1
	drop if dup[_n+1] == dup[_n] & dup[_n] != 0
	drop dup
}
/* Convert CIKCode to numeric */
gen CIKnumeric = real(subinstr(CIKCode, "0", "", 1))
drop CIKCode

rename CIKnumeric CIKCode

tempfile OwnerInformation
save `OwnerInformation' //has CIKCode and OwnTypeCode */
/*********************************************************************/

/**** Prepare TR mgrno Identifier for merging if Manager.dta available ****/
u "`path'\Data\WRDS-SEC\wrds_13f_link" if Flag >=2 //Flag >=2 are good matches between CIK and mgrno
rename cik CIKCode
drop coname Flag matchrate

*Drop duplicates, but keep first observation of duplicates
sort CIKCode
forval i = 1/4 {
	duplicates tag  CIKCode, gen(dup)
	replace dup = 1 if dup > 1
	drop if dup[_n+1] == dup[_n] & dup[_n] != 0
	drop dup
}

tempfile wrds_13f_link
save `wrds_13f_link' //has CIKCode and mgrno

/**** Prepare Manager.dta for merging ****/
u "`path'\Output\Manager.dta"
*Drop duplicates, but keep first observation of duplicates
sort mgrno rdate

forval i = 1/2 {
	duplicates tag  mgrno rdate, gen(dup)
	replace dup = 1 if dup > 1
	drop if dup[_n+1] == dup[_n] & dup[_n] != 0
	drop dup
}

*Reformat date
gen rdate_tq = qofd(rdate)
format rdate_tq %tq
drop rdate
rename rdate_tq rdate

tempfile Manager
save `Manager'
/*********************************************************************/

/* Load 13F Portfolio Holdings Data */
u "`path'\Data\SEC\Holdings\SEC_Sinkinson_1999-2012"
rename cik CIKCode

/**** Merge raw OwnerCode to Holdings data ****
merge m:1 CIKCode using `OwnerInformation'
*keep if _merge == 3
rename _merge mergeOwnInfo
drop CUSIP
drop OwnerCode */
/*********************************************************************/

/**** Merge mgrno to CIKCode ****/
merge m:1 CIKCode using `wrds_13f_link'
drop if _merge !=3 //Only keep observations that have an mgrno
drop _merge
/*********************************************************************/

/**** Merge cleaned Owner Type Code to Holdings Data ****/
merge m:1 rdate mgrno using `Manager', keepusing(mgrid typecode owntype type)
drop if _merge != 3 // Only keep observations that have a typecode
drop _merge
/*********************************************************************/

/**** Merge Stock Characteristics to Cusip ****/
rename cusip ncusip
drop if rdate < tq(2000q1)
merge m:1 rdate ncusip using "`path'\Output\StocksQ_clean"

*keep _merge as used in Holdings.do

/*********************************************************************/

label variable CIKCode "Manager Number used by SEC"
label variable rdate "Report Date"
label variable fdate "File Date of Report"
label variable shares "Number of shares held"

/*Reduce Dataset to include only necessary variables for 
	Holdings.do, Data0.do, Bins.ado, Data1.do
*/

*Data0 cleaning: Done here to reduce number of variables
gen nonMissing = 1 if profit<. & Gat<. & divA_be<. & LNbe<. & beta<. & LNret < . 
replace nonMissing = 0 if nonMissing == .
drop profit Gat divA_be LNbe beta LNret
label variable nonMissing "=1 if profit & Gat & divA_be & LNbe & beta & LNret <."


*Only keep important variables
keep fdate rdate mgrno mgrname mgrid shares ncusip shrout type typecode owntype permno date shrcd prc be nonMissing _merge

order fdate rdate mgrno mgrname mgrid type typecode owntype ncusip permno date shrcd prc shares shrout

sort rdate mgrno permno
compress

save "`path'\Output\SEC_Sinkinson_1999-2012_Final", replace 


/*
******************		File Description		******************
						----------------
						
							PART 2
						
This File merges SEC 13F Portfolio Holdings Data with stock characteristics & institution type codes. 
In doing so, it replaces the proprietary Thomson Reuters 13F Portfolio Data (s34) with the publically
available 13F Data from the SEC.
	
The 13F files are from 2013 to 2023 scraped by WRDS from the SEC Homepage.
In contrast to the proprietary TR Data, the distinction between the report date (rdate)
and the file date (fdate) is important here. The report date is the relevant merging date as 
this is the date for which the portfolio data is being reported. One can clearly see 
that the fdate is mostly 45 days after the report date, although sometimes the file date is even much later.

Relevant Files are 
	1) SEC WRDS 13F data
		---> Identifier: CIK. Contains: Portfolio Holdings data of stocks identified by CUSIP.
	
	2) StocksQ_clean from KY19
		---> Identifier: CUSIP. Contains: Stock Characteristics
	
	3) Thomson Reuters Global Ownership Owner Information
		---> Identifier: CIK. Contains: raw Owner Type Code
		
	4) Manager.dta 
		---> Identifier: mgrno. Contains: Cleaned Owner Type Code
	5) WRDS 13F Link File
		----> Links CIK Identifier to mgrno Identifier
	
How to get the relevant files:

	Thomson Reuters Global Ownership Owner Information was downloaded from WRDS (requires a subscription)
	
	WRDS 13F Link File was downloaded from WRDS (is available for free)
	
	StocksQ_clean is obtained from KY19's replication package (required CRSP/Compustat subscritpion).
	The file is cleaned in the sense that date formats and date variable names were edited for merging.
	
	Manager.dta is produced by the replication package of KY19 WHILE using the original s34 Thomson Reuters 13F data.
	
	It is possible to obtain the cleaned Owner Type Codes without Manager.dta
	by running KY19's Manager.do. 

Merging the Stock characteristics:
	The Stock characteristics can simply be merged through the cusip.
	
Merging the Owner Type Code:
	1) If Manager.dta is available:
		Complement CIK identifier with mgrno identifier via WRDS 13F Link File
		Merge Manager.dta via mgrno to datasets
	
	2) If Manager.dta NOT availalbe:
		Merge raw Thomson Reuters Owner Type Code via CIK from Thomson Reuters Global Ownership Owner Information.
		Execute the steps in Manager.do from KY19's replication package to clean the raw Owner Type Code. 
		
	
Important. The Stocks characteristics data must be lagged: 
	The Stock characteristics data (StocksQ) should be lagged for the following reason:
	Portfolio holdings are only reported at the end of the Quarter, but it is 
	likely that most trades happened at the beginning and middle of the quarter.
	Thus, at report date (rdate) t the institution most likely only had access
	to the stock characteristic data at time t-1 on which it based its holding decision.
	
 */
clear all
set more off
local path = "\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"



/**** Prepare Thomson Reuters Owner Type Code from Owner Information for merging ****
u "`path'\Output\Owner_info"
drop if CIKCode == ""
*Drop duplicates, but keep first observation of duplicates
sort CIKCode
forval i = 1/4 {
	duplicates tag CIKCode, gen(dup)
	replace dup = 1 if dup > 1
	drop if dup[_n+1] == dup[_n] & dup[_n] != 0
	drop dup
}
/* Convert CIKCode to numeric */
gen CIKnumeric = real(subinstr(CIKCode, "0", "", 1))
drop CIKCode

rename CIKnumeric CIKCode

tempfile OwnerInformation
save `OwnerInformation' //has CIKCode and OwnTypeCode */
/*********************************************************************/

/**** Prepare TR mgrno Identifier for merging if Manager.dta available ****/
u "`path'\Data\WRDS-SEC\wrds_13f_link" if Flag >=2 //Flag >=2 are good matches between CIK and mgrno
rename cik CIKCode
drop coname Flag matchrate

*Drop duplicates, but keep first observation of duplicates
sort CIKCode
forval i = 1/4 {
	duplicates tag  CIKCode, gen(dup)
	replace dup = 1 if dup > 1
	drop if dup[_n+1] == dup[_n] & dup[_n] != 0
	drop dup
}

tempfile wrds_13f_link
save `wrds_13f_link' //has CIKCode and mgrno

/**** Prepare Manager.dta for merging ****/
u "`path'\Output\Manager.dta"
*Drop duplicates, but keep first observation of duplicates
sort mgrno rdate

forval i = 1/2 {
	duplicates tag  mgrno rdate, gen(dup)
	replace dup = 1 if dup > 1
	drop if dup[_n+1] == dup[_n] & dup[_n] != 0
	drop dup
}

*Reformat date
gen rdate_tq = qofd(rdate)
format rdate_tq %tq
drop rdate
rename rdate_tq rdate

tempfile Manager
save `Manager'
/*********************************************************************/
/* Must Loop through eachy year as file to big for merge to complete */
forval i=2013/2023{
/* Load 13F Portfolio Holdings Data */
u "`path'\Data\WRDS-SEC\Holdings\SEC_WRDS_13F_2013-2023" if rdate >= tq(`i'q1) & rdate <= tq(`i'q4)
rename sshPrnamt shares

/* Convert CIKCode to numeric */
gen CIKnumeric = real(subinstr(cik, "0", "", 1))
drop cik
rename CIKnumeric CIKCode
/*********************************************************************/

/**** Merge mgrno to CIKCode ****/
merge m:1 CIKCode using `wrds_13f_link'
drop if _merge !=3 //Only keep observations that have an mgrno
drop _merge
/*********************************************************************/

/**** Merge cleaned Owner Type Code to Holdings Data ****/
merge m:1 rdate mgrno using `Manager', keepusing(mgrid typecode owntype type)
drop if _merge != 3 // Only keep observations that have a typecode
drop _merge
/*********************************************************************/

/**** Merge Stock Characteristics to Cusip ****/
rename cusip ncusip
drop if rdate < tq(2000q1)
merge m:1 rdate ncusip using "`path'\Output\StocksQ_clean"

*keep _merge as used in Holdings.do

/*********************************************************************/

label variable CIKCode "Manager Number used by SEC"
label variable rdate "Report Date"
label variable fdate "File Date of Report"
label variable shares "Number of shares held"

/*Reduce Dataset to include only necessary variables for 
	Holdings.do, Data0.do, Bins.ado, Data1.do
*/

*Data0 cleaning: Done here to reduce number of variables
gen nonMissing = 1 if profit<. & Gat<. & divA_be<. & LNbe<. & beta<. & LNret < . 
replace nonMissing = 0 if nonMissing == .
drop profit Gat divA_be LNbe beta LNret
label variable nonMissing "=1 if profit & Gat & divA_be & LNbe & beta & LNret <."

*Only keep important variables
keep fdate rdate mgrno mgrname mgrid shares ncusip shrout type typecode owntype permno date shrcd prc be nonMissing _merge

order fdate rdate mgrno mgrname mgrid type typecode owntype ncusip permno date shrcd prc shares shrout

sort rdate mgrno permno

compress

save "`path'\Output\SEC_WRDS_13F_`i'_Final", replace
}



/*
******************		File Description		******************
						----------------
						
							PART III
							
Append both intermediary datasets into one big dataset 

*/


clear all
set more off
local path = "\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"



u "`path'\Output\SEC_Sinkinson_1999-2012_Final"

forval i = 2013/2023 {
	append using "`path'\Output\SEC_WRDS_13F_`i'_Final"
}

sort rdate mgrno permno

drop if shares == 0 //Zero Holdings are constructed afterwards
save "`path'\Output\SEC_13F_Final", replace

/*
******************		File Description		******************
						----------------
						
							PART IV
							
Delete all intermediary files

*/

erase "`path\Output\SEC_Sinkinson_1999-2012_Final"

forval i = 2013/2023 {
	erase "`path'\Output\SEC_WRDS_13F_`i'_Final"
}
