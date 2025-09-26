/* Holdings.do (STATA)
	Construct Thomson Reuters stock holdings data.
	
	
	
	This file was adjusted to be able to process the 13F Data from the SEC */

clear all
set more off
set type double

**************************************

		**** ADJUST PATH ****

**************************************
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"
local forward = 11


/* Prepare s34 Stock Holdings */
u "`path'\Output\SEC_13F_Final"

/* Rescale Shares*/
replace shares = shares/1e6

/* Construct household sector */

sort rdate ncusip mgrno
by rdate ncusip: gen byte obs = 1+(_merge==3 & _n==1)

expand obs

sort rdate ncusip mgrno
by rdate ncusip: replace mgrname = "" if _merge==2 | (_merge==3 & _n==1)
by rdate ncusip: replace mgrno = 0 if _merge==2 | (_merge==3 & _n==1)
by rdate ncusip: replace mgrid = 0 if _merge==2 | (_merge==3 & _n==1)
by rdate ncusip: replace type = 0 if _merge==2 | (_merge==3 & _n==1)
by rdate ncusip: replace shares = . if _merge==2 | (_merge==3 & _n==1)

drop _merge obs

/* Fix miscoding of shares */

replace shares = min(shares,shrout) if mgrno>0

egen Tshares = sum(shares), by(rdate ncusip)

replace shares = max(shrout-Tshares,0) if mgrno==0
replace shares = shares/Tshares*shrout if Tshares>shrout

drop shrout Tshares

/* Construct zero holdings */

sort mgrno mgrid rdate 
by mgrno mgrid: gen byte obs = min(rdate[_N]-rdate,`forward')+1 

sort mgrno mgrid ncusip rdate 
by mgrno mgrid ncusip: replace obs = min(rdate[_n+1]-rdate,obs) 

expand obs

sort mgrno mgrid ncusip rdate 
by mgrno mgrid ncusip rdate: replace shares = 0 if _n>1 
by mgrno mgrid ncusip rdate: replace rdate = rdate+_n-1 

drop obs

/* Drop managers or stocks that do not exist */

egen Tshares1 = sum(shares), by(rdate mgrno)
egen Tshares2 = sum(shares), by(rdate ncusip)

drop if Tshares1==0 | Tshares2==0

drop Tshares*

/* Save data */

sort rdate mgrno ncusip 

save "`path'\Output\Holdings", replace

/* Duplicates in Holdings are cleaned in Python */
