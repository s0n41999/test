import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import GridSearchCV

#-----------------NASTAVENIA-----------------

st.set_page_config(layout="centered")

#------------------------------------------

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    predikcia()

def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    # Zjednodušenie názvov stĺpcov, ak je to potrebné
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','JPY=X', 'GBPUSD=X'])
moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','EURCHF=X', 'EURAUD=X','EURNZD=X', 'EURCAS=X', 'EURSEK=X', 'EURNOK=X', 'EURCZS=X'])
moznost = moznost.upper()
dnes = datetime.date.today()
start = dnes - datetime.timedelta(days=3650)
start_date = start
end_date = dnes

data = stiahnut_data(moznost, start_date, end_date)
scaler = StandardScaler()

# Zaistenie prístupu k stĺpcu 'Close' aj v prípade multi-indexu
close_column = [col for col in data.columns if 'Close' in col]
if close_column:
    data['Close'] = data[close_column[0]]

# Zobrazenie dát
st.write('Záverečný kurz')
st.line_chart(data['Close'])
st.header('Nedávne Dáta')
st.dataframe(data.tail(20))

# Výpočet a zobrazenie kĺzavých priemerov
st.header('Jednoduchý kĺzavý priemer za 50 dní')
datama50 = data.copy()
datama50['50ma'] = datama50['Close'].rolling(50).mean()
st.line_chart(datama50[['50ma', 'Close']])

st.header('Jednoduchý kĺzavý priemer za 200 dní')
datama200 = data.copy()
datama200['200ma'] = datama200['Close'].rolling(200).mean()
st.line_chart(datama200[['200ma', 'Close']])

# Spojenie dát pre kombinovaný graf
spojene_data = pd.concat([datama200[['200ma', 'Close']], datama50[['50ma']]], axis=1)

st.header('Jednoduchý kĺzavý priemer za 50 dní a 200 dní')
st.line_chart(spojene_data)

def predikcia():
    model_option = st.selectbox('Vyberte model', ['Lineárna Regresia', 'Regresor náhodného lesa', 'Regresor K najbližších susedov'])
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=5)
    pocet_dni = int(pocet_dni)
    if st.button('Predikovať'):
        if model_option == 'Lineárna Regresia':
            algoritmus = LinearRegression()
        elif model_option == 'Regresor náhodného lesa':
            # Optimalizácia hyperparametrov pre Random Forest
            param_grid = {
                'n_estimators': [50, 100, 150],
                'max_depth': [10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            rf = RandomForestRegressor(random_state=42)
            # Príprava dát pre GridSearchCV
            df = data[['Close']].copy()
            df['predikcia'] = df['Close'].shift(-pocet_dni)
            df.dropna(inplace=True)
            x = df[['Close']].values
            y = df['predikcia'].values
            x = scaler.fit_transform(x)
            grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
            grid_search.fit(x, y)
            algoritmus = grid_search.best_estimator_
            st.write(f"Najlepšie parametre: {grid_search.best_params_}")
        elif model_option == 'Regresor K najbližších susedov':
            # Optimalizácia hyperparametrov pre KNN
            param_grid = {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance']
            }
            knn = KNeighborsRegressor()
            # Príprava dát pre GridSearchCV
            df = data[['Close']].copy()
            df['predikcia'] = df['Close'].shift(-pocet_dni)
            df.dropna(inplace=True)
            x = df[['Close']].values
            y = df['predikcia'].values
            x = scaler.fit_transform(x)
            grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=3, n_jobs=-1, verbose=0)
            grid_search.fit(x, y)
            algoritmus = grid_search.best_estimator_
            st.write(f"Najlepšie parametre: {grid_search.best_params_}")

        vykonat_model(algoritmus, pocet_dni, model_option)

def vykonat_model(model, pocet_dni, model_name): 
    df = data[['Close']].copy()
    df['predikcia'] = df['Close'].shift(-pocet_dni)
    df.dropna(inplace=True)
    x = df[['Close']].values
    y = df['predikcia'].values

    # Škálovanie vstupných údajov
    x = scaler.fit_transform(x)

    # Chronologické rozdelenie dát
    train_size = int(len(x) * 0.8)
    x_trenovanie, x_testovanie = x[:train_size], x[train_size:]
    y_trenovanie, y_testovanie = y[:train_size], y[train_size:]

    # Trénovanie modelu
    model.fit(x_trenovanie, y_trenovanie)

    # Predikcia na testovacej množine
    predikcia = model.predict(x_testovanie)

    # Vyhodnotenie modelu
    rmse = np.sqrt(mean_squared_error(y_testovanie, predikcia))
    mae = mean_absolute_error(y_testovanie, predikcia)
    st.text(f'RMSE: {rmse}\nMAE: {mae}')

    # Predikcia na ďalšie dni
    posledne_hodnoty = x[-pocet_dni:]
    predikcia_forecast = model.predict(posledne_hodnoty)

    # Zobrazenie predikcií
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
