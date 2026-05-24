# Avast-CTU Public CAPEv2 Dataset
​
There is a limited amount of publicly available data to support research in malware analysis technology. Particularly, there are virtually no publicly available datasets generated from rich sandboxes such as Cuckoo/CAPE. The benefit of using dynamic sandboxes is the realistic simulation of file execution in the target machine and obtaining a log of such execution. The machine can be infected by malware hence there is a good chance of capturing the malicious behavior in the execution logs,  thus allowing researchers to study such behavior in detail. Although the subsequent analysis of log information is extensively covered in industrial cybersecurity backends, to our knowledge there has been only limited effort invested in academia to advance such log analysis capabilities using cutting edge techniques. We make this sample dataset available to support designing new machine learning methods for malware detection, especially for automatic detection of generic malicious behavior. The dataset has been collected in cooperation between Avast Software and Czech Technical University - AI Center (AIC).
​
The archive contains CAPEv2 reports of 48,976 malicious files. For each file, we provide the following metadata:
* sha256,
* classification to malware family,
* type of the malware,
* date of detection of the file.
  
There are 6 types of malware ( "banker", "trojan", "pws", "coinminer", "rat", "keylogger") and 10 malware families in the dataset:
​
| Malware Family Name | Number of Samples |
| ------------------- | ----------------- |
| Adload              | 704               |
| Emotet              | 14,429            |
| HarHar              | 655               |
| Lokibot             | 4,191             |
| njRAT               | 3,372             |
| Qakbot              | 4,895             |
| Swisyn              | 12,591            |
| Trickbot            | 4,202             |
| Ursnif              | 1,343             |
| Zeus                | 2,594             |
​
## Collection:
We publish execution logs collected from running malware samples identified as malware in Avast backend systems. Note that the executed files themselves are not part of the published dataset.
We collected the logs of the samples in the time period 07-09/2021 on the CAPEv2 instances operated at the Artificial Intelligence Center, at the Department of Computer Science, Faculty of Electrical Engineering, Czech Technical University.
​
## Authors:
* Branislav Bosansky (Avast / AIC, Dept. of Computer Science, FEE, CTU in Prague)
* Dominik Kouba (AIC, Dept. of Computer Science, FEE, CTU in Prague)
* Ondrej Manhal (AIC, Dept. of Computer Science, FEE, CTU in Prague)
* Thorsten Sick (Avast)
* Petr Somol (Avast)
* Viliam Lisy (Avast / AIC, Dept. of Computer Science, FEE, CTU in Prague)
* Jakub Kroustek (Avast)
