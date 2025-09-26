/* Data0.do (STATA)
	Merge institutional holdings and stock characteristics.
	by Ralph Koijen & Motohiro Yogo */

	
clear all
set more off
set type double

/* Load institutional holdings */
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

u "`path'\Output\Holdings_noDuplicates", clear //Alternative load Holdings.dta from DoFile 7b. But: Holdings.dta contains duplicates.
drop ncusip mgrname

/******* IF HOLDINGS NO DUPLICATES LOADED ***********************/
*Fix Dates if Holdings_noDuplicates was loaded from Python
format rdate %tc
gen rdate2 = rdate
gen year = year(dofc(rdate2))
gen month = month(dofc(rdate2))
gen quarter = ceil(month/3)
gen rdate3 = yq(year, quarter)
format rdate3 %tq
drop rdate rdate2 year month quarter
rename rdate3 rdate

format fdate %tc
gen fdate2 = fdate
gen year = year(dofc(fdate2))
gen month = month(dofc(fdate2))
gen quarter = ceil(month/3)
gen fdate3 = yq(year, quarter)
format fdate3 %tq
drop fdate fdate2 year month quarter
rename fdate3 fdate
/* ************************************************************ */

/* Define inside assets */
gen byte inside = inlist(shrcd,10,11) & nonMissing == 1 //& profit<. & Gat<. & divA_be<. & LNbe<. & beta<. & LNret<.
drop shrcd nonMissing

/* Define small institutions */

gen holding = prc*shares

egen aum = sum(holding), by(rdate mgrno)
egen outaum = sum(holding*!inside), by(rdate mgrno)

gen byte small = aum<10 | inlist(float(outaum/aum),0,1)

/* Combine small institutions with hh sector */

replace mgrno = 0 if small

egen Tshares = sum(shares), by(rdate mgrno permno)
egen Tholding = sum(holding), by(rdate mgrno permno)
egen Taum = sum(holding), by(rdate mgrno)
egen Toutaum = sum(holding*!inside), by(rdate mgrno)

foreach var of varlist shares holding aum outaum {
	replace `var' = T`var' if mgrno==0
}

keep if inside & !small

drop inside small Tshares Tholding Taum Toutaum

/* Construct variables */

egen int Xholding = count(holding), by(rdate mgrno)
egen int Nholding = sum(holding>0), by(rdate mgrno)

gen weight = holding/aum
gen outweight = outaum/aum

gen rweight = holding/outaum
gen LNrweight = ln(rweight)

egen cons = mean(LNrweight), by(rdate mgrno)

/* Label variables */

label var holding	"Amount held ($ million)"
label var aum		"Assets under management ($ million)"
label var outaum	"Outside assets under management ($ million)"

label var Xholding	"Count: holding"
label var Nholding	"Count: holding>0"
label var weight	"Portfolio weight"
label var outweight	"Outside portfolio weight"
label var rweight	"holding/outaum"
label var LNrweight	"Log: holding/outaum"
label var cons		"Mean: LNrweight"

*Merge book equity (be) as required in Data1 to construct instrument
merge m:1 rdate permno using "`path'\Output\StocksQ_rdate.dta", keepusing(be)
keep if _merge == 3
drop _merge

/*Save Data */
save "`path'\Output\Data0_noDuplicates", replace

