/* Summary.do (STATA)
	Summarize 13F institutions.
	by Ralph Koijen & Motohiro Yogo */


clear all
set more off
set type double

local forward = 11


/* Load data */
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"
u fdate mgrno mgrid type permno holding aum Xholding Nholding if holding>0 using "`path'\Output\Data0", clear

/* Construct variables */

gen byte period = floor((fdate-tq(1980q1))/20)+1

sort fdate mgrno permno
by fdate mgrno: gen _aum = aum if _n==1

/*
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
COMMAND NOT WORKING
egen byte Paum = xtile(_aum) if mgrno>0, n(10) by(fdate)

by fdate mgrno: replace Paum = Paum[1]
*/

egen Saum = sum(_aum), by(fdate)
replace Saum = 100*aum/Saum

/* Construct persistence of holdings */

sort mgrno mgrid fdate permno
by mgrno mgrid: gen _fdate = fdate-fdate[_n-1]

egen byte Dfdate = max(_fdate), by(fdate mgrno mgrid)
egen int first_mgrno = min(fdate), by(mgrno mgrid)
egen int first_permno = min(fdate), by(permno)

sort mgrno mgrid permno fdate

forval i = 1/`forward' {
	gen byte Iholding`i' = 0 if Dfdate<=`i' & fdate>=max(first_mgrno,first_permno)+`i'

	by mgrno mgrid permno: replace Iholding`i' = 100 if holding[_n-1]<. & fdate<=fdate[_n-1]+`i'
}

drop _fdate Dfdate first_mgrno first_permno

/* Persistence of holdings by AUM */

tabstat Iholding*, by(Paum) s(mean) format(%8.0f) not

/* Save data */

sort fdate mgrno permno
by fdate mgrno: keep if _n==1

tempfile Summary
save `Summary'


/* Summarize by date */

collapse (count) obs=aum (rawsum) Saum (median) aum Nholding Xholding (p90) aum90=aum Nholding90=Nholding Xholding90=Xholding if mgrno>0, by(fdate period) fast

/* Output table */

tabstat obs Saum aum aum90 Nholding Nholding90 Xholding Xholding90, by(period) s(mean) format(%8.0f) not

/* Summarize by institution type & date */

u `Summary', clear

collapse (count) obs=aum (rawsum) Saum (median) aum Nholding Xholding (p90) aum90=aum Nholding90=Nholding Xholding90=Xholding if mgrno>0, by(fdate period type) fast

/* Output table */

sort type
by type: tabstat obs Saum aum aum90 Nholding Nholding90 Xholding Xholding90, by(period) s(mean) format(%8.0f) not

