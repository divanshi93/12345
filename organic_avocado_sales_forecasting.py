def run_forecast(file_path):
    import pandas as pd
    import numpy as np
    import itertools
    import matplotlib.pyplot as plt
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from prophet import Prophet
    import statsmodels.api as sm
    import os
    
    # Read CSV file
    data = pd.read_csv(file_path)
    data['date'] = pd.to_datetime(data['date'])

    # Filter organic, Total US data
    my_data = data[(data['geography'] == 'Total U.S.') & (data['type'] == 'organic')][['date', 'total_volume']]
    my_data = my_data[(my_data['date'].dt.year != 2019) & (my_data['date'].dt.year != 2020)].set_index('date')
    my_data.drop(pd.Timestamp('2018-01-01'), inplace=True)
    my_data.rename(columns={'total_volume':'y'}, inplace=True)

    # Split train-test
    train = my_data[:181]
    test = my_data[181:]

    # Forecast using Exponential Smoothing (53-week seasonality)
    fcast_model = ExponentialSmoothing(train['y'], trend='additive', seasonal='multiplicative', seasonal_periods=53).fit()
    y_fcast = fcast_model.forecast(len(test))

    # Plot result
    plt.figure(figsize=(12,6))
    plt.plot(train.index, train['y'], label='Train')
    plt.plot(test.index, test['y'], label='Test')
    plt.plot(test.index, y_fcast, label='Forecast')
    plt.legend()
    plt.title('Triple Exponential Smoothing Forecast')
    forecast_path = os.path.join("static", "forecast.png")
    plt.savefig(forecast_path)
    plt.close()

    # Prophet Forecast
    df = train.reset_index().rename(columns={'date': 'ds'})
    m = Prophet(seasonality_mode='multiplicative')
    m.fit(df)
    future = m.make_future_dataframe(len(test), freq='W')
    forecast = m.predict(future)

    # Prophet plot
    fig1 = m.plot(forecast)
    fig1.savefig(os.path.join("static", "prophet_forecast.png"))
    plt.close(fig1)

    return "Forecast complete. Charts saved."