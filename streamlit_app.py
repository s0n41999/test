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
from streamlit_gsheets import GSheetsConnection
#-----------------NASTAVENIA-----------------

st.set_page_config(layout="centered")


#------------------------------------------
# Pripojenie ku Google Sheets
conn = st.experimental_connection("gsheets", type=GSheetsConnection)

# Čítanie údajov zo špecifikovaného sheetu
data = conn.read(spreadsheet_id=st.secrets["connections"]["gsheets"]["spreadsheet"], worksheet_name="predikcie")

# Zobrazenie načítaných údajov
st.write(data)






st.title('Predikcia časových radov vybraných valutových kurzov')

def main():
    predikcia()



def stiahnut_data(user_input, start_date, end_date):
    df = yf.download(user_input, start=start_date, end=end_date, progress=False)
    return df


moznost = st.selectbox('Zadajte menový tiker', ['EURUSD=X','JPY=X', 'GBPUSD=X'])
moznost = moznost.upper()
dnes = datetime.date.today()
start = dnes - datetime.timedelta(days=3650)
start_date = start
end_date = dnes


data = stiahnut_data(moznost, start_date, end_date)
scaler = StandardScaler()


st.write('Záverečný kurz')
st.line_chart(data.Close)
st.header('Nedávne Dáta')
st.dataframe(data.tail(20))


st.header('Jednoduchý kĺzavý priemer za 50 dní')
datama50=data 
datama50['50ma'] = datama50['Close'].rolling(50).mean()
st.line_chart(datama50[['50ma', 'Close']])

st.header('Jednoduchý kĺzavý priemer za 200 dní')
datama200=data 
datama200['200ma'] = datama200['Close'].rolling(200).mean()
st.line_chart(datama200[['200ma', 'Close']])

spojene_data = pd.merge(datama200[['200ma', 'Close']], datama50[['50ma']], left_index=True, right_index=True)

st.header('Jednoduchý kĺzavý priemer za 50 dní a 200 dní')
st.line_chart(spojene_data)



def dataframe():
    st.header('Nedávne dáta')
    st.dataframe(data.tail(10))


# Funkcia na pridanie predikcie do Google Sheets
def nacitat_do_google_sheets(predikcie, data_predicted, model_name):
    try:
        # Načítanie existujúcich údajov
        existing_data = conn.read(spreadsheet_id=st.secrets["connections"]["gsheets"]["spreadsheet"], worksheet_name=sheet_name)
        
        # Pridanie predikcií do tabuľky
        for index, row in data_predicted.iterrows():
            date_time = row['Deň']
            prediction = row['Predikcia']
            existing_data = existing_data.append({
                'Deň': str(date_time),
                'Predikcia': prediction,
                'Model': model_name
            }, ignore_index=True)

        # Zapísať dáta naspäť do Google Sheets
        conn.write(existing_data, spreadsheet_id=st.secrets["connections"]["gsheets"]["spreadsheet"], worksheet_name=sheet_name)
        st.success(f"Predikcie úspešne nahrané do sheetu {sheet_name}")
    except Exception as e:
        st.error(f"Nepodarilo sa nahrať predikcie: {e}")

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
        elif model == 'Regresor K najbližších susedov':
            algoritmus = KNeighborsRegressor()
            vykonat_model(algoritmus, pocet_dni)



def vykonat_model(model, pocet_dni): 
    df = data[['Close']]
    df['predikcia'] = data.Close.shift(-pocet_dni)
    x = df.drop(['predikcia'], axis=1).values
    x = scaler.fit_transform(x)
    x_predikcia = x[-pocet_dni:]
    x = x[:-pocet_dni]
    y = df.predikcia.values
    y = y[:-pocet_dni]

    #rozdelenie dát
    x_trenovanie, x_testovanie, y_trenovanie, y_testovanie = train_test_split(x, y, test_size=.2, random_state=7)
    # trénovanie modelu
    model.fit(x_trenovanie, y_trenovanie)
    predikcia = model.predict(x_testovanie)
    
    # predikcia na základe počtu dní
    predikcia_forecast = model.predict(x_predikcia)
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
    st.text(f'RMSE: {rmse} \
            \nMAE: {mean_absolute_error(y_testovanie, predikcia)}')
    
    if st.button('Načítať do Google Sheets'):
        nacitat_do_google_sheets('predikcie', data_predicted, model_name)  # Zapíšeme do "predikcie"


if __name__ == '__main__':
    main()
