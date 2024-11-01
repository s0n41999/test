from sklearn.model_selection import RandomizedSearchCV

def vykonat_model(model, pocet_dni, model_name): 
    df = data[['Close']].copy()
    df['predikcia'] = df['Close'].shift(-pocet_dni)
    df.dropna(inplace=True)
    x = df[['Close']].values
    y = df['predikcia'].values

    # Scale the features
    x = scaler.fit_transform(x)

    # Split data chronologically
    train_size = int(len(x) * 0.8)
    x_trenovanie, x_testovanie = x[:train_size], x[train_size:]
    y_trenovanie, y_testovanie = y[:train_size], y[train_size:]

    # Hyperparameter tuning for Random Forest and KNN using RandomizedSearchCV
    if model_name == 'Regresor náhodného lesa':
        param_distributions = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 5, 10],
            'min_samples_split': [2, 5, 10]
        }
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(x_trenovanie, y_trenovanie)
        model = random_search.best_estimator_
        st.write(f'Najlepšie hyperparametre: {random_search.best_params_}')
    elif model_name == 'Regresor K najbližších susedov':
        param_distributions = {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance']
        }
        random_search = RandomizedSearchCV(model, param_distributions, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42)
        random_search.fit(x_trenovanie, y_trenovanie)
        model = random_search.best_estimator_
        st.write(f'Najlepšie hyperparametre: {random_search.best_params_}')

    # Train the model
    model.fit(x_trenovanie, y_trenovanie)

    # Prediction on test set
    predikcia = model.predict(x_testovanie)

    # Evaluate the model
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
