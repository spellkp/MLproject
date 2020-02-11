# MLproject

Goal: 
-----

Predict the columns labeled "target1" and "target2" using the numerical and categorical attributes (only Column Z2 is categorical in nature) as input variables. For this problem, false positives are more costly than false negatives - certainty regarding predicted events is paramount and thus the accuracy of the model might best be scored using positive predictive value or lift at depth of 10-20%. Area under the ROC curve and Averaged Squared Error are also sound and common error metrics used in this application.

Target1 and Target2 are not mutually exclusive events. Each row could have neither event, both events, or one of the events. The two targets are related in some sense and researchers would be interested in anything that might inform them that one event might be more likely to happen than the other. So far the approach to that problem is to build two models and use the output from both models to decide which event is more likely to happen. 
