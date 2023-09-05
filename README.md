# Introuction

This GitHub repository hosts Python functions created for the Converge Project's forecasting component. To learn more about the project, please visit [here](https://arena.gov.au/projects/project-converge-act-distributed-energy-resources-demonstration-pilot/). The repository offers two primary functions: time-series forecasting for predicting real and reactive electrical power, and time-series disaggregation for breaking down net connection power into solar and demand components. YThe mathematical formulation and details of these functionalities can be found in [[1]](https://raw.githubusercontent.com/SeyyedMahdiNoori/converge_load_forecasting_data/main/Disaggregation.pdf) and [[2]](https://raw.githubusercontent.com/SeyyedMahdiNoori/Forecasting.pdf). Here, we provide a brief overview of the forecasting and disaggregation modules.

##### Forecasting

Most time-series forecasting approaches can be identfied depending on their uniques responses to the following four questions:
    
* What data is available for training the forecasting model? The answer to this question dictates the input to the forecasting approaches.

* How can the selected information be related to future estimations? This aspect is often referred to as the regression model in the literature.

*  What is an appropriate loss function to quantify the quality of the estimations? The answer to this question will determine the objective functions used in the forecasting approaches to train and optimise the parameters of the regression models.

* What is the optimal strategy for estimating a customer's load for multiple time steps into the future?

Our forecasting module allows users to select responses to these questions and construct various forecasting approaches. 

  



##### Disaggregation

Time-series disaggregation module explores way for behind-the-meter energy disaggregation to different sources using an aggregated measured value. When no extra information is available, and only the aggregated measurements of solar and demand are given, this problem is equivalent to an underdetermined system of linear equations with infinitely many solutions. Consequently, the most approaches introduce additional information to the problem of finding a unique solution that approximates the "true value" of solar and demand. Examples of these approaches are those using: weather data, geographically close solar exemplars and and reactive power. The disaggregation module offers seven techniques to disaggregate solar and demand from their aggregated value.

# Approaches and References
This package leverages a powerful combination of libraries, including **[skforecast](https://skforecast.org/0.9.1/index.html)**, **[tspiral](https://github.com/cerlymarco/tspiral)**, and **[scikit-learn](https://scikit-learn.org/stable/)**, to empower users in their forecasting endeavors. With this tool, you can flexibly select from various regression algorithms, such as linear regression, random forest, and XGBoost regression, tailored to different objectives like LSE, Ridge, and Lasso during model training. Moreover, you have the freedom to opt for diverse multi-step ahead algorithms like Recursive, Direct, Stacked, and Rectified. To enhance efficiency, parallel programming capabilities are included to accelerate calculations for time-series forecasting and disaggregation across multiple customers. Additionally, the package offers the capability to employ probabilistic techniques such as bootstrap and jackknife for generating predictions.


# Examples
Four example files are added in this package. The examples use [Ausgrid](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) and [NextGen](https://dl.acm.org/doi/abs/10.1145/3307772.3331017?casa_token=_QK2JaDahG8AAAAA:E3FwIUqQExbJDcHqOtc8684uzq8WI_eSEN4YokpMtU_pgkqZf5aMInKWTuvoPIlOSsh7MSKUZ3lP-g) publickly available data and show case some of the forecasting and disaggregation functionalities.


# Get in Touch
Feel free to send an email to **mahdi.noori@anu.edu.au**, if you have any question or suggestion.


