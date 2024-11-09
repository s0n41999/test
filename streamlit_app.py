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

moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','EURCHF=X', 'EURAUD=X','EURNZD=X', 'EURCAD=X', 'EURSEK=X', 'EURCZK=X'])
moznost = moznost.upper()
dnes = datetime.date(2024, 10, 1)
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
datama50 = data.copy()
datama50['50ma'] = datama50['Close'].rolling(50).mean()
st.line_chart(datama50[['50ma', 'Close']])

st.header('Jednoduchý kĺzavý priemer za 200 dní')
datama200 = data.copy()
datama200['200ma'] = datama200['Close'].rolling(200).mean()
st.line_chart(datama200[['200ma', 'Close']])


spojene_data = pd.concat([datama200[['200ma', 'Close']], datama50[['50ma']]], axis=1)

st.header('Jednoduchý kĺzavý priemer za 50 dní a 200 dní')
st.line_chart(spojene_data)


def dataframe():
    st.header('Nedávne dáta')
    st.dataframe(data.tail(10))



def predikcia():
    
    vyber_model = st.selectbox('Vyberte model', ['Lineárna Regresia', 'Regresor náhodného lesa', 'Regresor K najbližších susedov'])
    pocet_dni = int(st.number_input('Koľko dní chcete predpovedať?', value=5))

    if vyber_model == 'Lineárna Regresia':
        algoritmus = LinearRegression()
    elif vyber_model == 'Regresor náhodného lesa':
         algoritmus = RandomForestRegressor()
    elif vyber_model == 'Regresor K najbližších susedov':
         algoritmus = KNeighborsRegressor()

    
    if st.button('Predikovať'):
         # Vyberáme iba hodnoty zatváracej ceny
        data_ceny = data[['Close']]

        # Posúvame zatváracie ceny o stanovený počet dní na predikciu
        data_ceny['predikcia'] = data['Close'].shift(-pocet_dni)

        # Normalizujeme dáta pre model
        x_vstup = data_ceny.drop(columns=['predikcia']).values
        x_vstup = scaler.fit_transform(x_vstup)

        # Uchovávame posledných 'pocet_dni' na predikciu
        x_predikcia = x_vstup[-pocet_dni:]
        
        # Získavame hodnoty vstupných dát pre tréning
        x_trening = x_vstup[:-pocet_dni]
        
        # Extrahujeme predikčné hodnoty a obmedzíme na tréningovú množinu
        y_trening = data_ceny['predikcia'].values[:-pocet_dni]
        
        # Rozdelenie dát na tréningovú a testovaciu množinu
        train_size = int(len(x_trening) * 0.8)
        x_trenovanie, x_testovanie = x_trening[:train_size], x_trening[train_size:]
        y_trenovanie, y_testovanie = y_trening[:train_size], y_trening[train_size:]
        
        # Trénovanie modelu
        algoritmus.fit(x_trenovanie, y_trenovanie)  # Použitie 'algoritmus' na trénovanie
        predikcia = algoritmus.predict(x_testovanie)  # Predikcia s použitím 'algoritmus'
        
        # Predikcia na základe počtu dní
        predikcia_forecast = algoritmus.predict(x_predikcia)
        
        # Výpočet predikcie pre nasledujúce dni
        den = 1
        predikovane_data = []
        for i in predikcia_forecast:
            aktualny_datum = dnes + datetime.timedelta(days=den)
            st.text(f'Deň {den}: {i}')
            predikovane_data.append({'datum': aktualny_datum, 'predikcia': i})
            den += 1
        
        # Vytvorenie DataFrame s predikovanými dátami
        data_predicted = pd.DataFrame(predikovane_data)
        
        # Výpočet metriky chýb
        rmse = np.sqrt(np.mean((y_testovanie - predikcia) ** 2))
        mae = mean_absolute_error(y_testovanie, predikcia)
        st.text(f'RMSE: {rmse} \nMAE: {mae}')

        nazov_modelu = vyber_model.replace(' ', '_')  # Pre názov súboru odstránime medzery
        nazov_suboru = f'predikcia_{moznost}_{nazov_modelu}.csv'
        
        # Tlačidlo na stiahnutie dát s korektným delimitérom
        csv = data_predicted.to_csv(index=False, sep=';', encoding='utf-8')
        st.download_button(
            label="Stiahnuť predikciu ako CSV",
            data=csv,
            file_name=nazov_suboru,
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
