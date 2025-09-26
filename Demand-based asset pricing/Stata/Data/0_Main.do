* This STATA do-file runs all code in the "Data" folder.

* Merge institutional holdings and stock characteristics.

do 1_Data0

* Group institutions by type and aum. (Can be run in parallel in multiple STATA sessions.)

forval i = 1/8 {
	do 2_Bins_`i'
}

* Construct instrument for market equity.

do 3_Data1

* Summarize 13F institutions.

do 4_Summary
