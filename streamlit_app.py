import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

#-----------------SETTINGS-----------------
st.set_page_config(layout="centered")

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    predikcia()

def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','JPY=X', 'GBPUSD=X'])
moznost = moznost.upper()
dnes = datetime.date.today()
start = dnes - datetime.timedelta(days=1825)  # Limit data to last 5 years
start_date = start
end_date = dnes

data = stiahnut_data(moznost, start_date, end_date)
scaler = StandardScaler()

# Ensure the 'Close' column is accessible even if multi-indexed
close_column = [col for col in data.columns if 'Close' in col]
if close_column:
    data['Close'] = data[close_column[0]]

# Add moving averages and lagged features
data['50ma'] = data['Close'].rolling(50).mean()
data['200ma'] = data['Close'].rolling(200).mean()
data['lag_1'] = data['Close'].shift(1)
data['lag_5'] = data['Close'].shift(5)
data['lag_10'] = data['Close'].shift(10)
data.dropna(inplace=True)

# Plotting the data
st.write('Záverečný kurz')
st.line_chart(data['Close'])
st.header('Nedávne Dáta')
st.dataframe(data.tail(20))

st.header('Jednoduchý kĺzavý priemer za 50 dní a 200 dní')
st.line_chart(data[['Close', '50ma', '200ma']])

def predikcia():
    model_option = st.selectbox('Vyberte model', [
        'Lineárna Regresia', 
        'Regresor náhodného lesa', 
        'Regresor K najbližších susedov',
        'Gradient Boosting Regressor'
    ])
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=3)
    pocet_dni = int(pocet_dni)
    
    if st.button('Predikovať'):
        if model_option == 'Lineárna Regresia':
            algoritmus = LinearRegression()
            vykonat_model(algoritmus, pocet_dni, model_option)
        elif model_option == 'Regresor náhodného lesa':
            algoritmus = RandomForestRegressor(random_state=42)
            vykonat_model(algoritmus, pocet_dni, model_option)
        elif model_option == 'Regresor K najbližších susedov':
            algoritmus = KNeighborsRegressor()
            vykonat_model(algoritmus, pocet_dni, model_option)
        elif model_option == 'Gradient Boosting Regressor':
            algoritmus = GradientBoostingRegressor(random_state=42)
            vykonat_model(algoritmus, pocet_dni, model_option)

def vykonat_model(model, pocet_dni, model_name): 
    df = data[['Close', '50ma', '200ma', 'lag_1', 'lag_5', 'lag_10']].copy()
    df['predikcia'] = df['Close'].shift(-pocet_dni)
    df.dropna(inplace=True)
    
    x = df[['Close', '50ma', '200ma', 'lag_1', 'lag_5', 'lag_10']].values
    y = df['predikcia'].values

    # Scale features
    x = scaler.fit_transform(x)

    # Split data
    x_trenovanie, x_testovanie, y_trenovanie, y_testovanie = train_test_split(x, y, test_size=0.2, shuffle=False)

    # Optimized Randomized Search for RF & GB models
    if model_name == 'Regresor náhodného lesa':
        param_grid_rf = {
            'n_estimators': [30, 50, 70],
            'max_depth': [10, 15, 20],
            'min_samples_split': [5, 10],
            'min_samples_leaf': [2, 5]
        }
        grid_search = RandomizedSearchCV(model, param_grid_rf, cv=3, scoring='neg_mean_squared_error', n_iter=10)
        grid_search.fit(x_trenovanie, y_trenovanie)
        model = grid_search.best_estimator_
        st.write(f'Najlepšie hyperparametre: {grid_search.best_params_}')
    elif model_name == 'Gradient Boosting Regressor':
        param_grid_gb = {
            'n_estimators': [50, 100],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        grid_search = RandomizedSearchCV(model, param_grid_gb, cv=3, scoring='neg_mean_squared_error', n_iter=5)
        grid_search.fit(x_trenovanie, y_trenovanie)
        model = grid_search.best_estimator_
        st.write(f'Najlepšie hyperparametre: {grid_search.best_params_}')

    # Train the model
    model.fit(x_trenovanie, y_trenovanie)

    # Prediction on test set
    predikcia = model.predict(x_testovanie)

    # Evaluate the model
    rmse = np.sqrt(mean_squared_error(y_testovanie, predikcia))
    mae = mean_absolute_error(y_testovanie, predikcia)
    st.text(f'RMSE: {rmse}\nMAE: {mae}')

    # Forecast for next days
    posledne_hodnoty = x[-pocet_dni:]
    predikcia_forecast = model.predict(posledne_hodnoty)

    # Display predictions
    den = 1
    predikovane_data = []
    for i in predikcia_forecast:
        aktualny_datum = dnes + datetime.timedelta(days=den)
        predikovane_data.append({'Deň': aktualny_datum, 'Predikcia': i})
        den += 1
    data_predicted = pd.DataFrame(predikovane_data)
    st.dataframe(data_predicted)

if __name__ == '__main__':
    main()
