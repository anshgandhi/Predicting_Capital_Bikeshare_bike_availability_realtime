# Predicting Capital Bikeshare bike availability

Competition : Hippo Hacks 2018  

This is my submission to the Hippo Hacks 2018 Hackathon held at The George Washington University sponsored by Google.  

The current historical data provided by Capital Bikeshare consists of details of each individual trip, whereas the reason to go ahead with this project is to predict the availability of bikes at the stations during the selected time of the day.

The backbone of the app rests on 2 cronjobs running constantly.  
The first, downloads real-time data for all the bike dock stations (500 stations) in DC every 5 minutes from the Capital Bikeshare data feed. This data is fed into Google CloudSQL storage.

The second, runs two Machine Learning models (RandomForest Regressor and Neural Network) every hour on the entiredata residing in the Google CloudSQL storage. The script saves the weights it calculates every hour.


A Flask application is then deployed to read user input (Date, time, Location of station) to predict the number of available bikes at the specified station.

The application is deployed on Google Cloud Platform and for security purposes the application has been set to be accessible from within The George Washington University ip address ranges only.

Link : 35.196.136.99:5000 [http://35.196.136.99:5000/]

anshgandhi16@gmail.com