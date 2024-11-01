import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

#-----------------NASTAVENIA-----------------
st.set_page_config(layout="centered")

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    predikcia()

def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    # Flatten columns if necessary (handle multi-index case)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

# Možnosti výberu menového tikera
moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','JPY=X', 'GBPUSD=X'])
moznost = moznost.upper()
dnes = datetime.date.today()
start = dnes - datetime.timedelta(days=3650)
start_date = start
end_date = dnes

data = stiahnut_data(moznost, start_date, end_date)

# Ensure the 'Close' column is accessible even if multi-indexed
close_column = [col for col in data.columns if 'Close' in col]
if close_column:
    data['Close'] = data[close_column[0]]

st.write('Záverečný kurz')
st.line_chart(data.Close)
st.header('Nedávne Dáta')
st.dataframe(data.tail(20))

# Výpočet kĺzavých priemerov
st.header('Jednoduchý kĺzavý priemer za 50 dní')
datama50 = data.copy()
datama50['50ma'] = datama50['Close'].rolling(50).mean()
st.line_chart(datama50[['50ma', 'Close']])

st.header('Jednoduchý kĺzavý priemer za 200 dní')
datama200 = data.copy()
datama200['200ma'] = datama200['Close'].rolling(200).mean()
st.line_chart(datama200[['200ma', 'Close']])

# Merging 50ma and 200ma data for combined chart
spojene_data = pd.concat([datama200[['200ma', 'Close']], datama50[['50ma']]], axis=1)

st.header('Jednoduchý kĺzavý priemer za 50 dní a 200 dní')
st.line_chart(spojene_data)

def dataframe():
    st.header('Nedávne dáta')
    st.dataframe(data.tail(10))

# Pre spracovanie modelov
scaler = StandardScaler()
def predikcia():
    model = st.selectbox('Vyberte model', ['Lineárna Regresia', 'Regresor náhodného lesa', 'Regresor K najbližších susedov'])
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=5)
    pocet_dni = int(pocet_dni)
    if st.button('Predikovať'):
        if model == 'Lineárna Regresia':
            algoritmus = LinearRegression()
            vykonat_model(algoritmus, pocet_dni)
        elif model == 'Regresor náhodného lesa':
            algoritmus = RandomForestRegressor()
            vykonat_model(algoritmus, pocet_dni)
            algoritmus = RandomForestRegressor(n_estimators=100)
        elif model == 'Regresor K najbližších susedov':
            algoritmus = KNeighborsRegressor()
            vykonat_model(algoritmus, pocet_dni)
            algoritmus = KNeighborsRegressor(n_neighbors=5)

        vykonat_model(algoritmus, pocet_dni)

# Optimalizovaný model s TimeSeriesSplit a GridSearchCV
def vykonat_model(model, pocet_dni): 
    df = data[['Close']]
    df['predikcia'] = data.Close.shift(-pocet_dni)
    x = df.drop(['predikcia'], axis=1).values
    x = scaler.fit_transform(x)
    x_predikcia = x[-pocet_dni:]
    x = x[:-pocet_dni]
    y = df.predikcia.values
    df = data[['Close', '50ma', '200ma', 'stddev', 'momentum', 'RSI']].dropna()
    df['predikcia'] = df['Close'].shift(-pocet_dni)
    X = df.drop(['predikcia'], axis=1).values
    X = scaler.fit_transform(X)
    X_predikcia = X[-pocet_dni:]
    X = X[:-pocet_dni]
    y = df['predikcia'].values
    y = y[:-pocet_dni]

    #rozdelenie dát
    x_trenovanie, x_testovanie, y_trenovanie, y_testovanie = train_test_split(x, y, test_size=.2, random_state=7)
    # trénovanie modelu
    model.fit(x_trenovanie, y_trenovanie)
    predikcia = model.predict(x_testovanie)
    # Používame TimeSeriesSplit namiesto náhodného rozdelenia dát
    tscv = TimeSeriesSplit(n_splits=5)

    # predikcia na základe počtu dní
    predikcia_forecast = model.predict(x_predikcia)
    den = 1
    # Optimalizácia hyperparametrov (napr. v prípade KNN alebo RandomForest)
    if isinstance(model, KNeighborsRegressor):
        param_grid = {'n_neighbors': range(2, 10)}
        grid_search = GridSearchCV(model, param_grid, cv=tscv)
        grid_search.fit(X, y)
        model = grid_search.best_estimator_
    elif isinstance(model, RandomForestRegressor):
        param_grid = {'n_estimators': [50, 100, 200]}
        grid_search = GridSearchCV(model, param_grid, cv=tscv)
        grid_search.fit(X, y)
        model = grid_search.best_estimator_
    # Trénovanie modelu s TimeSeriesSplit
    model.fit(X, y)
    predikcia_forecast = model.predict(X_predikcia)
    predikovane_data = []
    den = 1
    col1, col2 = st.columns(2)

    with col1:
        for i in predikcia_forecast:
            aktualny_datum = dnes + datetime.timedelta(days=den)
            st.text(f'Deň {den}: {i}')
            predikovane_data.append({'Deň': aktualny_datum, 'Predikcia': i})
            den += 1

    with col2:
         data_predicted = pd.DataFrame(predikovane_data)
         st.dataframe(data_predicted)
    
    rmse = np.sqrt(np.mean((y_testovanie - predikcia) ** 2))
    st.text(f'RMSE: {rmse} \
            \nMAE: {mean_absolute_error(y_testovanie, predikcia)}')
        data_predicted = pd.DataFrame(predikovane_data)
        st.dataframe(data_predicted)

    # RMSE a MAE hodnoty
    st.text(f'RMSE: {np.sqrt(mean_squared_error(y, model.predict(X)))}')
    st.text(f'MAE: {mean_absolute_error(y, model.predict(X))}')

if __name__ == '__main__':
    main()
