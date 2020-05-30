# Using data from accelerometers to qualify how well people do exercise

## Summary
One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants is used.

Two prediction models, Decision Tree and Random Forest, have been built to predict the quality of the execution of the exercises using a training data set of more than 10k samples. As a result, Random Forest performed the best and was used to run predictions for 20 additional samples.

## Analysis
It can be seen in the HTML file.

## Conclusion
The prediction model based on Random Forest seems to be very adequeate to predict the classe variable using the measurements from the relevant sensors. With an accuracy of 99%, we are in good position to predict the quality of the execution of the 20 additional samples that have been given.
