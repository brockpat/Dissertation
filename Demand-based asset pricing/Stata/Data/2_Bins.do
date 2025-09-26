clear all
set more off
set type double

forval q = `=tq(2002q1)'(1)`=tq(2022q4)' { //Beginning and end of Data0.dta Time Series

/* Set parameters */
local path "\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"
local lag = 4

local Mholding = 1000	/* Minimum holdings for institution-level estimation */

/* Load data */

u rdate mgrno mgrid type permno holding aum Nholding if inlist(rdate,`q'-`lag',`q') & holding>0 using "`path'\Output\Data0_noDuplicates", clear

/* Construct lag variables */

sort mgrno mgrid permno rdate
qui by mgrno mgrid permno: gen Uholding = holding[_n-1] if rdate==rdate[_n-1]+`lag'

egen int UNholding = count(Uholding), by(rdate mgrno)

qui keep if rdate==`q'

/* Construct bins by aum */

gen byte Baum = 1

qui forval i = 1/6 {
	sum aum if type==`i' & Nholding<`Mholding'

	local Nbin = round(r(N)/(2*`Mholding'))	/* Number of bins */

	if `Nbin'>1 {
		pctile points = aum if type==`i' & Nholding<`Mholding', n(`Nbin')
		xtile bin = aum if type==`i', cut(points)

		replace Baum = bin if type==`i'

		drop points bin
	}
}

/* Construct bins by aum conditional on lagged holding */

gen byte UBaum = 1

qui forval i = 1/6 {
	sum aum if type==`i' & Uholding<. & UNholding<`Mholding'

	local Nbin = round(r(N)/(2*`Mholding'))	/* Number of bins */

	if `Nbin'>1 {
		pctile points = aum if type==`i' & Uholding<. & UNholding<`Mholding', n(`Nbin')
		xtile bin = aum if type==`i', cut(points)

		replace UBaum = bin if type==`i'

		drop points bin
	}
}

/* Construct overall bin */

egen int bin = group(type Baum)
egen int Ubin = group(type UBaum)

/* Label variables */

label var UNholding	"Count: Lagged holding"
label var Baum		"Bin: aum"
label var UBaum		"Bin: aum conditional on lagged holding"
label var bin		"Overall bin"
label var Ubin		"Overall bin conditional on lagged holding"

/* Save data */

sort rdate mgrno permno
qui by rdate mgrno: keep if _n==1

keep rdate mgrno type UNholding Baum UBaum bin Ubin

if `q'>`=tq(2002q1)' { //Make sure its first time period of Data0
	qui append using "`path'\Output\Bins"
}

sort rdate mgrno

qui save "`path'\Output\Bins", replace

}