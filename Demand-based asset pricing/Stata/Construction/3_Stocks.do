/* Stocks.do (STATA)
	Construct monthly & quarterly stock data.
	by Ralph Koijen & Motohiro Yogo */


clear all
set more off
set type double

**************************************

		**** ADJUST PATH ****

**************************************

/* Load CRSP Monthly Stock */
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

u "`path'\Output\Stocks_Monthly" if ldate>=tm(1963m6), clear

/* Merge beta */

merge 1:1 permno ldate using "`path'\Output\Beta", keepusing(beta) nogen keep(master match)

/* Winsorize variables */
*ssc install winsor2
winsor2 profit profitA accruals Gcsho Gat payA_be beta if inlist(shrcd,10,11), replace cuts(2.5 97.5) by(ldate)

winsor2 divA_be if inlist(shrcd,10,11), replace cuts(0 97.5) by(ldate)

/* Save data */

sort permno ldate

save "`path'\Output\StocksM", replace


/* Construct quarterly returns */

sort permno ldate
by permno: gen LNret = 0 if ldate==ldate[_n-3]+3 & prc<. & prc[_n-3]<.
by permno: gen LNretx = 0 if ldate==ldate[_n-3]+3 & prc<. & prc[_n-3]<.

forval i = 0/2 {
	by permno: replace LNret = LNret+ln(1+ret[_n-`i']) if ldate==ldate[_n-`i']+`i' & ret[_n-`i']<.
	by permno: replace LNretx = LNretx+ln(1+retx[_n-`i']) if ldate==ldate[_n-`i']+`i' & retx[_n-`i']<.
}

gen LNretd = LNret-LNretx

/* Construct annual returns */

sort permno ldate
by permno: gen LNretA = 0 if ldate==ldate[_n-12]+12 & prc<. & prc[_n-12]<.
by permno: gen LNretxA = 0 if ldate==ldate[_n-12]+12 & prc<. & prc[_n-12]<.

forval i = 0/11 {
	by permno: replace LNretA = LNretA+ln(1+ret[_n-`i']) if ldate==ldate[_n-`i']+`i' & ret[_n-`i']<.
	by permno: replace LNretxA = LNretxA+ln(1+retx[_n-`i']) if ldate==ldate[_n-`i']+`i' & retx[_n-`i']<.
}

gen LNretdA = LNretA-LNretxA

drop ldate daret daretx

/* Keep quarter-end */

keep if mod(month(date),3)==0

replace ret = exp(LNret)-1
replace retx = exp(LNretx)-1

/* Construct variables */

gen int fdate = qofd(date)

/* Construct lag variables */

sort permno fdate
by permno: gen _me = me[_n-1] if fdate==fdate[_n-1]+1

gen _meA = .

forval i = 1/4 {
	by permno: replace _meA = me[_n-`i'] if fdate==fdate[_n-`i']+4
}

/* Sample criteria */

keep if fdate>=tq(1980q1)

keep if ncusip~=""
keep if prc<.
keep if shrout>0 & shrout<.

/* Label variables */

format %tq fdate

label var LNret		"Log: 1+ret"
label var LNretx	"Log: 1+retx"
label var LNretd	"LNret-LNretx"
label var LNretA	"Log annual: 1+ret"
label var LNretxA	"Log annual: 1+retx"
label var LNretdA	"LNretA-LNretxA"

label var fdate		"File Date"

label var _me		"Lag: me"
label var _meA		"Lag annual: me"

/* Save data */

sort permno fdate

save  "`path'\Output\StocksQ", replace
