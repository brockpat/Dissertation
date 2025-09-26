/* Data1.do (STATA)
	Construct instrument for market equity.
	by Ralph Koijen & Motohiro Yogo */

clear all
set more off
set type double

local forward = 11
local cutoff = 95
 
/* Load data */
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

u "`path'\Output\Data0_noDuplicates", clear

/* Merge bins */ 

merge m:1 rdate mgrno using "`path'\Output\Bins", keepusing(UNholding bin Ubin) nogen keep(master match)

/* Construct persistence of holdings */

egen int first_mgrno = min(rdate), by(mgrno mgrid)
egen int first_permno = min(rdate), by(permno)

gen byte Iholding = 0 if holding>0 & rdate>max(first_mgrno,first_permno)
replace Iholding = 100 if rdate==tq(2002q1) //What to do with this? in KY19 it's 1980q1

sort mgrno mgrid permno rdate


forval i = 1/`forward' {
	by mgrno mgrid permno: replace Iholding = 100 if Iholding==0 & holding[_n-`i']>0 & holding[_n-`i']<. & rdate<=rdate[_n-`i']+`forward'
}

egen Mholding = mean(Iholding), by(rdate mgrno)

drop first_mgrno first_permno Iholding

/* Construct instrument */

egen Iweight = count(weight), by(rdate mgrno)
replace Iweight = (mgrno>0)/(1+Iweight)	/* Exclude household sector */

egen demand1 = sum(aum*Iweight*(Mholding>=`cutoff' & Mholding<.)), by(rdate permno)
gen demand2 = aum*Iweight*(Mholding>=`cutoff' & Mholding<.)	/* Exclude own holding */

gen IVme = ln(demand1-demand2)

drop Iweight demand1 demand2

/* Construct instrument (book equity weights) */

egen Iweight = sum(be), by(rdate mgrno)
replace Iweight = be*(mgrno>0)/Iweight	/* Exclude household sector */

egen demand1 = sum(aum*Iweight*(Mholding>=`cutoff' & Mholding<.)), by(rdate permno)
gen demand2 = aum*Iweight*(Mholding>=`cutoff' & Mholding<.)	/* Exclude own holding */

gen IVmeBE = ln(demand1-demand2)

drop Iweight Mholding demand1 demand2

/* Label variables */

label var IVme		"Instrument for LNme"
label var IVmeBE	"Instrument for LNme (book equity weights)"

/* Save data */

sort rdate mgrno permno

save "`path'\Output\Data1", replace


/* Save characteristics & instrument 

u rdate mgrno permno profit Gat divA_be beta LNbe LNme IVme IVmeBE if mgrno==0 using "`path'\Output\Data1", clear

drop mgrno

sort rdate permno

save "`path'\Output\IV", replace

*/
