# pyMatAna
The repository contains the python codes for analyzing the system response matrix.

## PPDF Analysis
PPDF_Analysis folder
### Angular projection analysis
```angular-projection.py``` is used for running in batch.
An example of performing a batch analysis is as followed:
```shell
fnamelist=$(ls ../data/)
for fname in $fnamelist; do python angular-projection.py $fname;done
```
```angular-projection.ipynb``` is a jupyter notebook. It is used for interactive running and showing the analysis process.