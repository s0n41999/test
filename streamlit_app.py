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
import requests
import feedparser

#-----------------NASTAVENIA-----------------

st.set_page_config(layout="centered")

#------------------------------------------

st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    zobraz_spravy_v_sidebar()
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
    model_options = {
        'Lineárna Regresia': LinearRegression(),
        'Regresor náhodného lesa': RandomForestRegressor(),
        'Regresor K najbližších susedov': KNeighborsRegressor()
    }
    
    model = st.selectbox('Vyberte model', list(model_options.keys()))
    pocet_dni = st.number_input('Koľko dní chcete predpovedať?', value=5)
    pocet_dni = int(pocet_dni)
    
    if st.button('Predikovať'):
        algoritmus = model_options.get(model)

        # Príprava dát pre "lagged" predikciu
        df = data[['Close']].copy()
        
        # Pridáme oneskorené hodnoty (lag features) ako prediktory
        for lag in range(1, pocet_dni + 1):
            df[f'lag_{lag}'] = df['Close'].shift(lag)

        df.dropna(inplace=True)

        # Vytvorenie vstupných a výstupných hodnôt
        x = df.drop(['Close'], axis=1).values
        y = df['Close'].values

        # Rozdelenie dát na tréning a testovanie (na základe časového sledu)
        train_size = int(len(x) * 0.8)
        x_trenovanie, x_testovanie = x[:train_size], x[train_size:]
        y_trenovanie, y_testovanie = y[:train_size], y[train_size:]

        # Trénovanie modelu
        algoritmus.fit(x_trenovanie, y_trenovanie)

        # Predikcia na testovacej množine
        predikcia = algoritmus.predict(x_testovanie)

        # Predikcia budúcich hodnôt (na základe posledných známych hodnôt)
        posledne_data = x[-1].reshape(1, -1)
        predikcia_forecast = []
        
        for _ in range(pocet_dni):
            buduca_hodnota = algoritmus.predict(posledne_data)[0]
            predikcia_forecast.append(buduca_hodnota)
            
            # Aktualizujeme "lag" hodnoty na ďalšiu predikciu
            posledne_data = np.roll(posledne_data, -1)
            posledne_data[0, -1] = buduca_hodnota

        # Výpis predikcií
        den = 1
        predikovane_data = []
        for i in predikcia_forecast:
            aktualny_datum = dnes + datetime.timedelta(days=den)
            st.text(f'Deň {den}: {i}')
            predikovane_data.append({'datum': aktualny_datum, 'predikcia': i})
            den += 1

        data_predicted = pd.DataFrame(predikovane_data)

        # Hodnotenie modelu
        rmse = np.sqrt(np.mean((y_testovanie - predikcia) ** 2))
        mae = mean_absolute_error(y_testovanie, predikcia)
        st.text(f'RMSE: {rmse} \nMAE: {mae}')

        # Hodnotenie modelu na budúcich predikovaných hodnotách (rolling forecast evaluation)
        future_rmse = np.sqrt(np.mean((y_testovanie[:pocet_dni] - predikcia_forecast) ** 2))
        future_mae = mean_absolute_error(y_testovanie[:pocet_dni], predikcia_forecast)
        st.text(f'RMSE (Future Rolling Forecast): {future_rmse} \nMAE (Future Rolling Forecast): {future_mae}')

        # Button to download prediction data with the correct delimiter
        csv = data_predicted.to_csv(index=False, sep=';', encoding='utf-8')
        st.download_button(
            label="Stiahnuť predikciu ako CSV",
            data=csv,
            file_name=f'predikcia_{moznost}.csv',
            mime='text/csv'
        )

def zobraz_spravy_v_sidebar():
    st.sidebar.header('Aktuálne Správy súvisiace s Menovým Trhom :chart_with_upwards_trend:')
    st.sidebar.markdown('---')
    # Použitie RSS feedu pre načítanie finančných správ z Investing.com - Forex News sekcia
    feed_url = 'https://www.investing.com/rss/news_1.rss'  # RSS kanál zameraný na Forex News od Investing.com
    feed = feedparser.parse(feed_url)

    if len(feed.entries) > 0:
        for entry in feed.entries[:15]: 
            st.sidebar.subheader(entry.title)
            if hasattr(entry, 'summary'):
                st.sidebar.write(entry.summary)
            st.sidebar.write(f"[Čítať viac]({entry.link})")
            st.sidebar.markdown('---')  # Pridanie oddeľovacej čiary medzi správami
    else:
        st.sidebar.write('Nenašli sa žiadne správy.')

if __name__ == '__main__':
    main()
