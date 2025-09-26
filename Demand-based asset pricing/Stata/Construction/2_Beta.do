/* Beta.do (STATA)
	Estimate rolling beta.
	by Ralph Koijen & Motohiro Yogo */


clear all
set more off
set type double

**************************************

		**** ADJUST PATH ****

**************************************
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"


/* Load CRSP Monthly Stock data */

u permno date ldate daret if ldate>=tm(1958m6) using "`path'\Output\Stocks_Monthly", clear

/* Merge Fama-French factors */

merge m:1 date using "`path'\Data\Fama French/Fama-French Factors/Fama-French Factors Monthly", nogen keep(match)

/* Construct variables */

gen daret_rf = daret-rf

/* Estimate beta */

gen beta = .

qui forval m = `=tm(2000m1)'/`=tm(2023m12)' {
    gen byte window = inrange(ldate,`m'-60,`m'-1)
    egen byte obs = count(daret_rf) if window, by(permno)

    egen Mmktrf = mean(mktrf) if daret_rf<. & window & obs>=24, by(permno)

    gen xx = (mktrf-Mmktrf)^2 if daret_rf<. & window & obs>=24
    gen xy = (mktrf-Mmktrf)*daret_rf if daret_rf<. & window & obs>=24

    egen Mxx = mean(xx), by(permno)
    egen Mxy = mean(xy), by(permno)

	replace beta = Mxy/Mxx if ldate==`m'

    drop window obs Mmktrf xx xy Mxx Mxy
}

drop daret_rf

/* Sample criteria */

keep if ldate>=tm(1963m6)

keep if beta<.

/* Label variables */

label var beta "Market beta"

/* Save data */

sort permno ldate

save "`path'\Output\Beta", replace
