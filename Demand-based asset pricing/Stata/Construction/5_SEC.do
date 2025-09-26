/* Manager.do (STATA)
	Construct SEC Form ADV data.
	by Ralph Koijen & Motohiro Yogo */

clear all
set more off
set type double

**************************************

		**** ADJUST PATH ****

**************************************
local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"

/* Loop through all ADV files */
*Note: SEC.txt contains all filenames in `path'\Data\SEC\Form ADV\Registered. This
*file was generated outside of Stata

! dir "`path'\Data\SEC\Form ADV\Registered\*.dta" /a-d /b > "`path'\Data\SEC\Form ADV\Registered\SEC.txt"
file open SEC using "`path'\Data\SEC\Form ADV\Registered\SEC.txt", read

/*
		!!!!!!!!!		CAUTION			!!!!!!!!!
The !dir command might not work depending on the STATA version and user access rights.
The error will be "file SEC not found" once the loop starts.
The loop is very simple and can be coded outside of Stata:
	Loop through every dta file, 
	only load the primary business name, legal name and the SEC effective date
	rename variables
	reformat the date (can also be done afterwards).
	Append all files 
	This completes the loop.
*/

qui forval i = 1/170 { //Number of files is 169, for safety one more was added

	/* Load file */

	file read SEC filename

	if r(eof) {
		continue, break
	}
	local path"\\ntsamba1.server.uni-frankfurt.de\pbrock\Desktop\KY19 Replication"
	u Primary_Business_Name Legal_Name *Effective_Date using "`path'\Data\SEC\Form ADV/Registered/`filename'", clear

	/* Rename variables */

	cap rename Effective_Date SEC_Status_Effective_Date
	cap rename Status_Effective_Date SEC_Status_Effective_Date

	/* Fix variables */

	cap confirm string var SEC_Status_Effective_Date

    if !_rc {
		gen int date = date(SEC_Status_Effective_Date,"MDY")
		drop SEC_Status_Effective_Date
		rename date SEC_Status_Effective_Date
	}

	/* Save data */

	if `i'>1 {
		append using SEC
	}

	save "`path'\Output\SEC", replace
}

file close SEC

/* Stack Primary_Business_Name & Legal_Name */

drop if SEC_Status_Effective_Date==.

sort Primary_Business_Name Legal_Name SEC_Status_Effective_Date
by Primary_Business_Name Legal_Name: keep if _n==1

expand 2 if Primary_Business_Name~=Legal_Name

sort Primary_Business_Name Legal_Name
by Primary_Business_Name Legal_Name: gen mgrname = Primary_Business_Name if _n==1
by Primary_Business_Name Legal_Name: replace mgrname = Legal_Name if _n==2

drop Primary_Business_Name Legal_Name

/* Format name */

replace mgrname = ustrto(ustrnormalize(mgrname,"nfd"),"ascii",2)

replace mgrname = subinstr(mgrname,"!"," ",.)
replace mgrname = subinstr(mgrname,"|"," ",.)
replace mgrname = subinstr(mgrname,""," ",.)
replace mgrname = subinstr(mgrname,`"""',"",.)
replace mgrname = subinstr(mgrname,","," ",.)
replace mgrname = subinstr(mgrname,".","",.)
replace mgrname = subinstr(mgrname,"/"," ",.)

replace mgrname = subinstr(mgrname,"%"," & ",.)
replace mgrname = subinstr(mgrname,"&"," & ",.)
replace mgrname = subinstr(mgrname,"+"," & ",.)

replace mgrname = regexr(mgrname,"(\(|\[).*($|\)|\])"," ")
replace mgrname = regexr(mgrname,"-+","-")

replace mgrname = upper(trim(itrim(mgrname)))

/* Keep unique name */

drop if regexm(mgrname,"^(|-)$")

sort mgrname SEC_Status_Effective_Date
by mgrname: keep if _n==1

/* Construct variables */

sort mgrname

gen int Uid = _n

/* Save data */

compress

save "`path'\Output\SEC", replace

/* Load s34 Manager */

u mgrname using "`path'/Data\Thomson Reuters/13F\s34 Manager", clear

duplicates drop

/* Construct variables */

sort mgrname

gen int id = _n

/* Merge SEC Form ADV */

reclink2 mgrname using "`path'\Output\SEC", idmaster(id) idusing(Uid) gen(score) minscore(.95) manytoone

keep if _merge==3

drop id Uid _merge

/* Keep unique observation */

sort mgrname Umgrname
by mgrname: keep if _n==1

/* Label variables */

label var score						"Matching score"
label var SEC_Status_Effective_Date	"SEC Status Effective Date"

/* Save data */

sort mgrname

save "`path'\Output\SEC", replace