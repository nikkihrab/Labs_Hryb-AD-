import os
import glob
import datetime
import urllib.request
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

# --- Завантаження даних ---
def download_data(province_id, year1=1981, year2=2024):
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    else:
        existing_files = [f for f in os.listdir(data_folder) if f.startswith(f'vhi_id__{province_id}__')]
        if existing_files:
            return  # Уже є файл — пропускаємо

    url = f"https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?country=UKR&provinceID={province_id}&year1={year1}&year2={year2}&type=Mean"
    with urllib.request.urlopen(url) as response:
        data = response.read()

    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f'vhi_id__{province_id}__{current_datetime}.csv'
    file_path = os.path.join('data', filename)

    with open(file_path, 'wb') as out:
        out.write(data)

# Завантажуємо дані для всіх 27 областей
for i in range(1, 28):
    download_data(i)

# --- Обробка CSV-файлів у єдиний DataFrame ---
def create_data_frame(folder_path):
    csv_files = glob.glob(folder_path + "/*.csv")
    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
    frames = []

    for file in csv_files:
        region_id = int(file.split('__')[1])
        df = pd.read_csv(file, header=1, names=headers)
        df.at[0, 'Year'] = df.at[0, 'Year'][9:]  # Видаляємо 'year' зі значення
        df = df.drop(df.index[-1])  # Останній рядок — порожній
        df = df.drop(df.loc[df['VHI'] == -1].index)  # Фільтруємо некоректні значення
        df = df.drop('empty', axis=1)
        df.insert(0, 'region_id', region_id, True)
        df['Week'] = df['Week'].astype(int)
        frames.append(df)

    result = pd.concat(frames).drop_duplicates().reset_index(drop=True)
    result = result.loc[(result.region_id != 12) & (result.region_id != 20)]  # Пропускаємо Крим та Луганськ
    result = result.replace({'region_id': {
        1: 22, 2: 24, 3: 23, 4: 25, 5: 3, 6: 4, 7: 8, 8: 19, 9: 20, 10: 21,
        11: 9, 13: 10, 14: 11, 15: 12, 16: 13, 17: 14, 18: 15, 19: 16, 21: 17,
        22: 18, 23: 6, 24: 1, 25: 2, 26: 6, 27: 5}})
    return result

df = create_data_frame('./data')

reg_id_name = {
    1: 'Вінницька', 2: 'Волинська', 3: 'Дніпропетровська', 4: 'Донецька', 5: 'Житомирська',
    6: 'Закарпатська', 7: 'Запорізька', 8: 'Івано-Франківська', 9: 'Київська', 10: 'Кіровоградська',
    11: 'Луганська', 12: 'Львівська', 13: 'Миколаївська', 14: 'Одеська', 15: 'Полтавська',
    16: 'Рівенська', 17: 'Сумська', 18: 'Тернопільська', 19: 'Харківська', 20: 'Херсонська',
    21: 'Хмельницька', 22: 'Черкаська', 23: 'Чернівецька', 24: 'Чернігівська', 25: 'Республіка Крим'
}

# --- Streamlit Інтерфейс ---
st.title("Лабораторна 3: Аналіз індексу вегетаційного здоров’я (VHI)")

parameter = st.selectbox("Оберіть параметр:", ["VCI", "TCI", "VHI"])
region_id = st.selectbox("Оберіть область:", sorted(df['region_id'].unique()), format_func=lambda x: reg_id_name.get(x, str(x)))
years_interval = st.text_input("Інтервал років (наприклад 2005-2020):", "1982-2024")
weeks_interval = st.text_input("Інтервал тижнів (наприклад 1-3):", "1-3")

if st.button("Оновити"):
    year_start, year_end = map(int, years_interval.split('-'))
    week_start, week_end = map(int, weeks_interval.split('-'))

    filtered = df[
        (df['Year'].astype(int).between(year_start, year_end)) &
        (df['Week'].between(week_start, week_end)) &
        (df['region_id'] == region_id)
    ][['Year', 'Week', parameter]]

    st.subheader("📊 Таблиця результатів")
    st.dataframe(filtered)

    if not filtered.empty:
        st.subheader("📈 Графік")
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))

        # Теплова карта
        pivot = filtered.pivot(index='Year', columns='Week', values=parameter)
        sns.heatmap(pivot, cmap='inferno', annot=True, ax=axes[0])
        axes[0].set_title(f"Теплова карта {parameter} по роках і тижнях")

        # Лінійна діаграма
        for year in filtered['Year'].unique():
            data_year = filtered[filtered['Year'] == year]
            sns.lineplot(data=data_year, x='Week', y=parameter, ax=axes[1], label=str(year))

        axes[1].set_title(f"Динаміка {parameter} за тижнями для області: {reg_id_name[region_id]}")
        axes[1].set_xlabel("Тиждень")
        axes[1].set_ylabel(parameter)

        st.pyplot(fig)
    else:
        st.warning("⚠️ Немає даних для вибраного інтервалу.")