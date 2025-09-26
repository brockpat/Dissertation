/* Manager.do (STATA)
	Construct Thomson Reuters manager data.
	by Ralph Koijen & Motohiro Yogo */


clear all
set more off
set type double

**************************************

		**** ADJUST PATH ****

**************************************
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

/* Load s12 Type 5 Table */

u fdate mgrco mgrcocd using "`path'/Data\Thomson Reuters/13F/s12type5", clear

/* Reformat Date of s12type5 if not already done
gen test = date(fdate, "YMD")
format test %td
drop fdate
rename test fdate
order fdate mgrco mgrcocd
*/

duplicates drop

/* Rename variables */

rename mgrcocd mgrno

/* Save data */

tempfile s12type5
save `s12type5'

/* Load Thomson Reuters Owner Information */

u ownercode ownername owntypecode cikcode if cikcode~=. using "`path'\Data\Thomson Reuters\Ownership\Owner_Information.dta", clear

/* Rename variables */

rename ownername mgrname
rename cikcode cik

/* Fix variables */

replace mgrname = upper(mgrname)

/* Construct variables */

gen byte owntype = 6
replace owntype = 1 if inlist(owntypecode,101,302)
replace owntype = 2 if inlist(owntypecode,108)
replace owntype = 3 if inlist(owntypecode,106,107,113,402)
replace owntype = 4 if inlist(owntypecode,401)
replace owntype = 5 if inlist(owntypecode,110,114)

/* Drop duplicates */

sort mgrname cik owntype ownercode
by mgrname cik: keep if _n==1

/* Save data */

tempfile Owner
save `Owner'


/* Load WRDS 13F Mapping Table */
u mgrno mgrname cik Flag if Flag>=2 using "`path'/Data\WRDS-SEC/wrds_13f_link", clear

drop Flag

/* Construct variables */

sort mgrno mgrname

gen int id = _n

/* Merge Thomson Reuters Owner Information */

destring cik, replace
*search reclink2 to be able to install it
reclink2 mgrname cik using `Owner', idmaster(id) idusing(ownercode) gen(score) required(cik) minscore(.6) manytoone

keep if _merge==3

drop _merge

/* Drop duplicates */

sort mgrno mgrname owntype id
by mgrno mgrname: keep if _n==1

drop id

/* Save data */

tempfile wrds_13f_link
save `wrds_13f_link'


/* Load s34 Manager */
u fdate mgrname mgrno typecode rdate if fdate>=td(31mar2002) using "`path'/Data\Thomson Reuters/13F/s34 Manager", clear

sort mgrno fdate rdate
by mgrno fdate: keep if _n==1

/* Construct variables */

sort mgrno fdate
by mgrno: gen byte change = qofd(fdate)>qofd(fdate[_n-1])+2 & mgrname~=mgrname[_n-1]
by mgrno: gen byte mgrid = sum(change)

drop change

/* Fix typecode after December 1998 */

sort mgrno mgrid fdate
by mgrno mgrid: replace typecode = typecode[_n-1] if typecode[_n-1]<. & fdate>=td(31dec1998)

gsort mgrno mgrid -fdate
by mgrno mgrid: replace typecode = typecode[_n-1] if typecode[_n-1]<.

/* Merge s12 Type 5 Table */

gen mgrco = substr(mgrname,1,19)

merge 1:1 fdate mgrno mgrco using `s12type5', gen(mf) keep(master match)

gsort mgrno mgrid -mf fdate
by mgrno mgrid: replace mf = mf[1]

drop mgrco

/* Merge WRDS 13F Mapping Table */

merge m:1 mgrno mgrname using `wrds_13f_link', keepusing(owntype) nogen keep(master match)

sort mgrno mgrid fdate
by mgrno mgrid: replace owntype = owntype[_n-1] if owntype[_n-1]<.

gsort mgrno mgrid -fdate
by mgrno mgrid: replace owntype = owntype[_n-1] if owntype[_n-1]<.

/* Merge SEC Form ADV */

merge m:1 mgrname using "`path'\Output\SEC", keepusing(mgrname) gen(ia) keep(master match)

gsort mgrno mgrid -ia fdate
by mgrno mgrid: replace ia = ia[1]

/* Merge list of pension funds */

merge m:1 mgrno mgrid using "`path'\Output\Pension Funds_KY", gen(pf) keep(master match)

/* Fix typecode for investment advisors */

replace typecode = 4 if typecode==5 & ia==3

drop ia

/* Construct institution type */

gen byte type = 6
replace type = 1 if typecode==1
replace type = 2 if typecode==2
replace type = 3 if inrange(typecode,3,4)
replace type = 4 if inrange(typecode,3,5) & mf==3
replace type = 5 if inrange(typecode,3,5) & pf==3

replace type = 1 if type==6 & owntype==1
replace type = 2 if type==6 & owntype==2
replace type = 3 if type==6 & owntype==3
replace type = 4 if type==6 & owntype==4
replace type = 5 if type==6 & owntype==5

drop mf pf

/* Label variables */

label var mgrid		"Manager ID"
label var owntype	"Owner type (TR Ownership)"
label var type		"Institution type"

label define type_label 0 "Households" 1 "Banks" 2 "Insurance companies" 3 "Investment advisors"4 "Mutual funds" 5 "Pension funds" 6 "Other"

label val owntype type type_label

/* Save data */

sort fdate mgrno
gen test = date(rdate, "YMD")
format test %td
drop rdate
rename test rdate

save "`path'\Output\Manager", replace