This folder contains the data splits from v2 of Social Bias Frames / Social Bias Inference Corpus.
Simply read in the file using pandas:
```python
trndf = pd.read_csv("SBIC.v2.trn.csv")
```

Each line in the file contains the following fields (in order):
- _whoTarget_: group vs. individual target
- _intentYN_: was the intent behind the statement to offend
- _sexYN_: is the post a sexual or lewd reference
- _sexReason_: free text explanations of what is sexual
- _offensiveYN_: could the post be offensive to anyone
- _annotatorGender_: gender of the MTurk worker 
- _annotatorMinority_: whether the MTurk worker identifies as a minority
- _sexPhrase_: part of the post that references something sexual
- _speakerMinorityYN_: whether the speaker was part of the same minority group that's being targeted
- _WorkerId_: hashed version of the MTurk workerId
- _HITId_: id that uniquely identifies each post
- _annotatorPolitics_: political leaning of the MTurk worker
- _annotatorRace_: race of the MTurk worker
- _annotatorAge_: age of the MTurk worker
- _post_: post that was annotated
- _targetMinority_: demographic group targeted
- _targetCategory_: high-level category of the demographic group(s) targeted
- _targetStereotype_: implied statement
- _dataSource_: source of the post (`t/...`: means Twitter, `r/...`: means a subreddit)

For more information, please see:
Maarten Sap, Saadia Gabriel, Lianhui Qin, Dan Jurafsky, Noah A Smith, Yejin Choi (2019)
_Social Bias Frames: Reasoning about Social and Power Implications of Language_