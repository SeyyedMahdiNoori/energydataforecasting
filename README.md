# Introuction

In this repository, you can find the functions required for load forecasting in the Converge project. 

This repository contains the following forecasting methods:

* A forecasting approach, using the **skforecast** library, that is based on a [Recursive multi-step forecasting](https://joaquinamatrodrigo.github.io/skforecast/0.4.3/quick-start/introduction-forecasting.html#recursive-multi-step-forecasting) algorithm. This implementation outputs a forecasted value for aggregated solar generation and demand for a desired number of days with a granularity of 30 minutes for each *nmi*.

* A forecasting approach, using the **skforecast** library, that is based on a [Prediction intervals](https://joaquinamatrodrigo.github.io/skforecast/0.4.3/notebooks/prediction-intervals.html)(A prediction interval defines the interval within which the true value of a time-series variable can be expected to be found with a given probability) algorithm. This implementation outputs probabilitic values of expected value, upper bound and lower for aggregated solar generation and demand for a desired number of days with a granularity of 30 minutes for each *nmi*.



