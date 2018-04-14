# Predicting Capital Bikeshare bike availability

Competition : Hippo Hacks 2018  

This is my submission to the Hippo Hacks 2018 at The George Washington University sponsored by Google.
The app has cronjobs running which constantly downloads real-time data for all the dock stations (500 stations) in DC every 5 minutes from the Capital Bikeshare data feed. This data is then fed into a MySQL server.

A neural network model using tensorflow is then trained every-one hour on the data residing in the MySQL server, hence providing with updated weights every one hour.

A Flask application is then deployed to read user input (Date, time, Location of station) to predict the number of available bikes at the specified station.

The application deployed on Google Cloud Platform is only accessible from The George Washington University ip address range.