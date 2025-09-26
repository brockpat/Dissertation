/* Stocks_Monthly.do (STATA)
	Prepare CRSP & Compustat data.
	by Ralph Koijen & Motohiro Yogo */

clear all
set more off
set type double

**************************************

		**** ADJUST PATH ****

**************************************
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

/* Load Compustat Quarterly  */
u lpermno linkenddt datadate fyearq fqtr fyr ajexq actq atq ceqq cheq cogsq cshoq dlcq lctq ltq pstkq revtq seqq txditcq xintq xsgaq using "`path'\Data\CRSP\CRSP_Compustat_Merged_Quarterly.dta"
/* Rename variables */

foreach var in fyear ajex act at ceq che cogs csho dlc lct lt pstk revt seq txditc xint xsga {
	rename `var'q `var'
}

/* Fix variables */
gen x = mofd(linkenddt)
replace x = .e if x == .
drop linkenddt
gen linkenddt = .e
label var linkenddt "Last Effective Date of Link"
replace linkenddt = x
drop x
gen x = mofd(datadate)
replace x = .e if x == .
drop datadate
gen datadate = .e
label var datadate "Data Date"
replace datadate = x
drop x

order lpermno linkenddt datadate fyear fqtr fyr ajex act at ce che cogs csho dlc lct lt pstk revt se txditc xint xsga

foreach var of varlist act-xsga {
	egen M`var' = median(`var'), by(lpermno datadate)	/* Duplicates from changes in fyr */

	replace `var' = M`var'

	drop M`var'
}

replace csho = csho*ajex	/* Split-adjusted shares outstanding */

recode cogs xint xsga (.=0) if cogs<. | xint<. | xsga<.
recode txditc (.=0)

replace seq = ceq+pstk if seq==.
replace seq = at-lt if seq==.

replace at = lt+seq if at==.
replace lt = at-seq if lt==.

drop ajex

/* Annually aggregate flow variables */

sort lpermno fyr datadate

foreach var of varlist cogs revt txditc xint xsga {
	by lpermno fyr: gen S`var' = `var' if datadate==datadate[_n-3]+9
	
	forval i = 1/3 {
		by lpermno fyr: replace S`var' = S`var'+`var'[_n-`i'] if datadate==datadate[_n-`i']+3*`i'
	}

	replace `var' = S`var'

	drop S`var'
}

/* Construct variables */

gen int ldate = datadate+6

gen be = seq+txditc-pstk
replace be = . if be<=0

gen operating = act-che-lct+dlc

gen at_be = at/be

gen profit = (revt-cogs-xsga-xint)/be	/* Fama-French definition */
gen profitA = (revt-cogs)/at			/* Novy-Marx definition */

sort lpermno fyr fqtr fyear
by lpermno fyr fqtr: gen accruals = (operating-operating[_n-1])/be if fyear==fyear[_n-1]+1
by lpermno fyr fqtr: gen Gcsho = ln(csho/csho[_n-1]) if fyear==fyear[_n-1]+1
by lpermno fyr fqtr: gen Gat = ln(at/at[_n-1]) if fyear==fyear[_n-1]+1

foreach var of varlist be-Gat {
	egen M`var' = median(`var'), by(lpermno datadate)	/* Duplicates from changes in fyr */

	replace `var' = M`var'

	drop M`var'
}

/* Keep unique observation by lpermno & datadate (drop duplicates from changes in fyr) */

sort lpermno datadate fyr
by lpermno datadate: keep if _n==_N

/* Sample criteria */

keep if be<. & profit<. & Gat<.

/* Save data */

sort lpermno datadate

save "`path'\Output\Compustat", replace


/* Load Compustat Annual */

u gvkey lpermno linkenddt datadate fyear fyr ajex pddur act at ceq che cogs csho dlc lct lt pstk pstkl pstkrv revt seq txditc xint xsga sich if pddur==12 using "`path'\Data\CRSP\CRSP_Compustat_Merged_Annual", clear

drop pddur

/* Fix variables */

gen x = mofd(linkenddt)
replace x = .e if x == .
drop linkenddt
gen linkenddt = .e
label var linkenddt "Last Effective Date of Link"
replace linkenddt = x
drop x
gen x = mofd(datadate)
replace x = .e if x == .
drop datadate
gen datadate = .e
label var datadate "Data Date"
replace datadate = x
drop x

replace csho = csho*ajex	/* Split-adjusted shares outstanding */

recode cogs xint xsga (.=0) if cogs<. | xint<. | xsga<.
recode txditc (.=0)

replace seq = ceq+pstk if seq==.
replace seq = at-lt if seq==.

replace at = lt+seq if at==.
replace lt = at-seq if lt==.

drop ajex

/* Construct variables */

gen int ldate = datadate+6

gen preferred = pstkrv
replace preferred = pstkl if preferred==.
replace preferred = pstk if preferred==.

gen be = seq+txditc-preferred
replace be = . if be<=0

gen operating = act-che-lct+dlc

gen at_be = at/be

gen profit = (revt-cogs-xsga-xint)/be	/* Fama-French definition */
gen profitA = (revt-cogs)/at			/* Novy-Marx definition */

sort lpermno fyr fyear
by lpermno fyr: gen accruals = (operating-operating[_n-1])/be if fyear==fyear[_n-1]+1
by lpermno fyr: gen Gcsho = ln(csho/csho[_n-1]) if fyear==fyear[_n-1]+1
by lpermno fyr: gen Gat = ln(at/at[_n-1]) if fyear==fyear[_n-1]+1

drop preferred

/* Merge Compustat Quarterly */

merge 1:1 lpermno datadate using "`path'\Output\Compustat", nogen

/* Fill missing SIC code */

sort lpermno datadate
by lpermno: replace sich = sich[_n-1] if sich==.

/* Expand to monthly sample */

sort lpermno ldate
by lpermno: gen byte obs = min(ldate[_n+1]-ldate,12)

expand obs

sort lpermno ldate
by lpermno ldate: replace ldate = ldate+_n-1

drop obs

/* Sample criteria */

keep if ldate<=linkenddt

local year = 2024
keep if inrange(ldate,tm(1963m6),tm(`year'm12))

/* Label variables */

rename lpermno permno
rename sich siccd

format %tm linkenddt datadate ldate

label var ldate		"CRSP link date"
label var be		"Book equity ($ million)"
label var operating	"Operating working capital"
label var at_be		"Assets to Book equity"
label var profit	"Operating profit to Book equity"
label var profitA	"Gross profit to Assets"
label var accruals	"Change in operating working capital to Book equity"
label var Gcsho		"Log growth: Shares outstanding"
label var Gat		"Log growth: Assets"

/* Save data */

compress

sort permno ldate

save "`path'\Output\Compustat", replace

/*
##############################################################################
##############################################################################
							CAUTION	

# We don't have the S&P Index Constituents data yet
/* Load S&P Index Constituents */

u gvkey from thru spmi
	if spmi=="10"
	using "`path'/Compustat/`year'/Index Constituents", clear

/* Fix variables */

replace from = mofd(from)
replace thru = mofd(min(thru,td(31dec`year')))

/* Expand to monthly sample */

gen int obs = thru-from+1

expand obs

sort gvkey from
by gvkey from: gen int ldate = from+_n-1

drop from thru obs

duplicates drop

/* Sample criteria */

keep if ldate<=tm(`year'm12)

/* Label variables */

format %tm ldate

label var ldate	"CRSP link date"

/* Save data */

sort gvkey ldate

save "`path'\Output\SPIndex", replace
##############################################################################
##############################################################################
*/


/* Load CRSP Monthly Stock */

u permno date shrcd exchcd siccd ncusip dlretx dlret prc ret shrout retx if inlist(shrcd,10,11,12,18) & inrange(exchcd,1,3) using "`path'\Data\CRSP\CRSP_Monthly_Stock", clear


duplicates drop

/* Fix variables */

recode siccd (0=.)
recode prc (-99999=.)	/* Missing code */

replace prc = abs(prc)

replace shrout = shrout/1e3

/* Construct variables */

gen int ldate = mofd(date)

/* Merge Compustat */

merge 1:1 permno ldate using "`path'\Output\Compustat", keepusing(gvkey datadate siccd be at_be profit profitA accruals Gcsho Gat) update replace

drop if _merge==2
drop _merge


/*##############################################################################
S&P 500 CONSTITUENTS NOT AVAILABLE

/* Merge S&P index constituents */

merge m:1 gvkey ldate using SPIndex,
	nogen keep(master match)

/* Merge Fama-French industry classification */

merge m:1 siccd using "`path'/Fama French/Industry Definitions/Stata/SIC12"

replace Industry_Number = 12 if _merge==1
replace Industry_Name = "Other" if _merge==1

drop if _merge==2
drop _merge
##############################################################################
##############################################################################
*/

/* Construct variables */

gen me = prc*shrout

sort permno ldate

gen meA = .

forval i = 1/17 {
	by permno: replace meA = me[_n-`i'] if ldate[_n-`i']==datadate
}

gen be_meA = be/meA
/*
##############################################################################
S&P 500 CONSTITUENTS NOT AVAILABLE
gen byte Ispmi = spmi=="10"

drop spmi
##############################################################################
##############################################################################
*/

/* Construct momentum return */

sort permno ldate
by permno: gen Cret = 0 if ldate==ldate[_n-12]+12 & prc[_n-1]<. & prc[_n-12]<.

forval i = 1/11 {
	by permno: replace Cret = Cret+ln(1+ret[_n-`i']) if ret[_n-`i']<.
}

/* Construct cumulative factor to adjust price */

gen fac = .

sort permno ldate

forval i = 1/11 {
	by permno: replace fac = float((1+retx)*prc[_n-`i']/prc) if fac==.
}

by permno: replace fac = 1 if fac==. & sum(prc<.)>=1

by permno: gen cfac = 1 if sum(prc<.)==1
by permno: replace cfac = cfac[_n-1]*fac if cfac==.

/* Construct dividends and payout */

gen div = .
gen pay = .

sort permno ldate

forval i = 1/11 {
	by permno: replace div = (ret-retx)*prc[_n-`i']/fac if div==. //ret, retx & prc are loaded from the Data
	by permno: replace pay = (ret-(me/me[_n-`i']-1))*prc[_n-`i']/fac if pay==.
}

recode div pay (.=0) if prc<.

gen divA = div
gen payA = pay

forval i = 1/11 {
	by permno: replace divA = divA+div[_n-`i']*cfac[_n-`i']/cfac if ldate==ldate[_n-`i']+`i' & div[_n-`i']<.
	by permno: replace payA = payA+pay[_n-`i']*cfac[_n-`i']/cfac if ldate==ldate[_n-`i']+`i' & pay[_n-`i']<.
}

gen divA_be = .
gen payA_be = .

forval i = 1/17 {
	by permno: replace divA_be = divA[_n-`i']*shrout[_n-`i']/be if ldate[_n-`i']==datadate
	by permno: replace payA_be = payA[_n-`i']*shrout[_n-`i']/be if ldate[_n-`i']==datadate
}

/* Construct delisting adjusted return */

gen daret = ret if ret<. & dlret==.
replace daret = dlret if ret==. & dlret<.
replace daret = (1+ret)*(1+dlret)-1 if ret<. & dlret<.

gen daretx = retx if retx<. & dlretx==.
replace daretx = dlretx if retx==. & dlretx<.
replace daretx = (1+retx)*(1+dlretx)-1 if retx<. & dlretx<.

drop dlretx dlret

/* Construct log variables */

gen LNprc = ln(prc)
gen LNshrout = ln(shrout)
gen LNbe = ln(be)
gen LNme = ln(me)
gen LNmeA = ln(meA)
gen LNbe_meA = ln(be_meA)
gen LNcfac = ln(cfac)

/* Label variables */

format %tm ldate

label var me		"Market equity ($ million)"
label var meA		"Market equity on datadate ($ million)"
label var be_meA	"Book equity to Market equity on datadate"
/*##############################################################################
S&P 500 CONSTITUENTS NOT AVAILABLE
label var Ispmi		"Dummy: S&P 500 index"
##############################################################################
*/
label var Cret		"Cumulative log return from month t-12 to t-1"

label var fac		"Factor to adjust price"
label var cfac		"Cumulative factor to adjust price"

label var div		"Dividend per share"
label var pay		"Payout per share"
label var divA		"Annual dividend per share"
label var payA		"Annual payout per share"
label var divA_be	"Annual dividend to Book equity"
label var payA_be	"Annual payout to Book equity"

label var daret		"Delisting adjusted return"
label var daretx	"Delisting adjusted return without dividends"

label var LNprc		"Log: prc"
label var LNshrout	"Log: shrout"
label var LNbe		"Log: be"
label var LNme		"Log: me"
label var LNmeA		"Log: meA"
label var LNbe_meA	"Log: be_meA"
label var LNcfac	"Log: cfac"

/* Save data */

compress
sort permno ldate
save "`path'\Output\Stocks_Monthly", replace

