import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(layout="centered")

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    predikcia()

def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X', 'JPY=X', 'GBPUSD=X'])
moznost = moznost.upper()
dnes = datetime.date.today()
start = dnes - datetime.timedelta(days=3650)
start_date = start
end_date = dnes

data = stiahnut_data(moznost, start_date, end_date)
scaler = StandardScaler()

close_column = [col for col in data.columns if 'Close' in col]
if close_column:
    data['Close'] = data[close_column[0]]

st.write('Záverečný kurz')
st.line_chart(data['Close'])
st.header('Nedávne Dáta')
st.dataframe(data.tail(20))

st.header('Jednoduchý kĺzavý priemer za 50 dní')
data['50ma'] = data['Close'].rolling(50).mean()
st.line_chart(data[['50ma', 'Close']])

st.header('Jednoduchý kĺzavý priemer za 200 dní')
data['200ma'] = data['Close'].rolling(200).mean()
st.line_chart(data[['200ma', 'Close']])

st.header('Jednoduchý kĺzavý priemer za 50 dní a 200 dní')
spojene_data = data[['50ma', '200ma', 'Close']]
st.line_chart(spojene_data)

def predikcia():
    model_option = st.selectbox('Vyberte model', ['Lineárna Regresia', 'Regresor náhodného lesa', 'Regresor K najbližších susedov'])
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=5)
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

def vykonat_model(model, pocet_dni, model_name): 
    df = data[['Close']].copy()  # Vyberáme len 'Close' stĺpec
    df['predikcia'] = df['Close'].shift(-pocet_dni)
    df.dropna(inplace=True)
    x = df[['Close']].values
    y = df['predikcia'].values

    x = scaler.fit_transform(x)
    train_size = int(len(x) * 0.8)
    x_trenovanie, x_testovanie = x[:train_size], x[train_size:]
    y_trenovanie, y_testovanie = y[:train_size], y[train_size:]

    # Ak model vyžaduje hyperparameter tuning
    if model_name == 'Regresor náhodného lesa':
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(x_trenovanie, y_trenovanie)
        model = grid_search.best_estimator_
        st.write(f'Najlepšie hyperparametre: {grid_search.best_params_}')
    elif model_name == 'Regresor K najbližších susedov':
        param_grid = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
        grid_search = GridSearchCV(model, param_grid, cv=3, scoring='neg_mean_squared_error')
        grid_search.fit(x_trenovanie, y_trenovanie)
        model = grid_search.best_estimator_
        st.write(f'Najlepšie hyperparametre: {grid_search.best_params_}')

    model.fit(x_trenovanie, y_trenovanie)

    predikcia = model.predict(x_testovanie)
    rmse = np.sqrt(mean_squared_error(y_testovanie, predikcia))
    mae = mean_absolute_error(y_testovanie, predikcia)
    st.text(f'RMSE: {rmse}\nMAE: {mae}')

    posledne_hodnoty = x[-pocet_dni:]
    predikcia_forecast = model.predict(posledne_hodnoty)

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
