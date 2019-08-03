# Telstra Network Disruptions

Kaggle Telstra Competition
Solution entered after competition had closed - just for practice
Score: 0.52908

https://www.kaggle.com/c/telstra-recruiting-network/overview


Approach used:

•	Based on the data explanation “Each row in the main dataset (train.csv, test.csv) represents a location and a time point. They are identified by the "id" column, which is the key "id" used in other data files.” It was assumed that ‘id’ referred to the time point.  This was then used to link and group the remaining features in the other files
•	Created ‘examine_data.py’ to examine all data files to get an overview of available data and possible features
•	Created ‘feature_ extraction.py’ to extract features from data files
•	This involved 3 of the ‘log’ files from the dataset (severity, event and resource) and splitting them as separate features based on their feature ids such as severity_type 1, severity_type 2. Basic count and frequency metrics by id were also calculated 
•	Location and log features were treated slightly differently as they seemed to provide richer data. Where possible count, max, min, mean and median features were calculated. An ordering column was added based on the log features ordered by id. Finally the features locations were ranked in ascending and descending order and by relative rank based on location id and log order. 
•	All the extracted features (~500) were then output to a .csv file called ‘features_extracted.csv’
•	Created the file ‘feature_imp.py’ in which an XGBoost algorithm was used to rank the features in terms of importance and this was output to another.csv file ('feature_importance.csv'). In the final model this ranking was used to eliminate roughly half of the features generated
•	Finally, wrote a ‘csv_submission.py ‘file to run the model including 10 fold cross validation and create a .csv file to submit to Kaggle
•	Again this involved running an XGBoost algorithm and testing that the logloss was acceptable
