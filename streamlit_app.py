import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import datetime
from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error  

#-----------------NASTAVENIA-----------------

st.set_page_config(layout="centered")

#------------------------------------------

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    predikcia()

def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    # Flatten columns if necessary (handle multi-index case)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ['_'.join(col).strip() for col in df.columns.values]
    return df

moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','EURCHF=X', 'EURAUD=X','EURNZD=X', 'EURCAD=X', 'EURSEK=X', 'EURNOK=X', 'EURCZK=X'])
moznost = moznost.upper()
dnes = datetime.date.today()
start = dnes - datetime.timedelta(days=3650)
start_date = start
end_date = dnes

data = stiahnut_data(moznost, start_date, end_date)
scaler = StandardScaler()

# Ensure the 'Close' column is accessible even if multi-indexed
close_column = [col for col in data.columns if 'Close' in col]
if close_column:
    data['Close'] = data[close_column[0]]

# Plotting the data
st.write('Záverečný kurz')
st.line_chart(data['Close'])
st.header('Nedávne Dáta')
st.dataframe(data.tail(20))

# Calculating and plotting moving averages
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

def predikcia():
    model = st.selectbox('Vyberte model', ['Lineárna Regresia', 'Regresor náhodného lesa', 'Regresor K najbližších susedov'])
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=5)
    pocet_dni = int(pocet_dni)
    if st.button('Predikováť'):
        if model == 'Lineárna Regresia':
            algoritmus = LinearRegression()
        elif model == 'Regresor náhodného lesa':
            algoritmus = RandomForestRegressor()
        elif model == 'Regresor K najbližších susedov':
            algoritmus = KNeighborsRegressor()
        
        # Vykonanie modelu
        df = data[['Close']]
        df['predikcia'] = data.Close.shift(-pocet_dni)
        x = df.drop(['predikcia'], axis=1).values
        x = scaler.fit_transform(x)
        x_predikcia = x[-pocet_dni:]
        x = x[:-pocet_dni]
        y = df.predikcia.values
        y = y[:-pocet_dni]

        # Rozdelenie dát
        train_size = int(len(x) * 0.8)
        x_trenovanie, x_testovanie = x[:train_size], x[train_size:]
        y_trenovanie, y_testovanie = y[:train_size], y[train_size:]

        # Trénovanie modelu
        algoritmus.fit(x_trenovanie, y_trenovanie)
        predikcia = algoritmus.predict(x_testovanie)

        # Predikcia na základe počtu dní
        predikcia_forecast = algoritmus.predict(x_predikcia)
        den = 1
        predikovane_data = []
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
        st.text(f'RMSE: {rmse} \nMAE: {mean_absolute_error(y_testovanie, predikcia)}')

if __name__ == '__main__':
    main()
