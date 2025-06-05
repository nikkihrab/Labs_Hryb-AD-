import pandas as pd
import os
import urllib.request
import datetime
import glob
import matplotlib.pyplot as plt
import seaborn as sns
from ipywidgets import interact, widgets

# === STEP 1: Download VHI data ===
def download_data(province_id, year1=1981, year2=2024):
    data_folder = 'data'
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    else:
        existing_files = [f for f in os.listdir(data_folder) if f.startswith(f'vhi_id__{province_id}__')]
        if existing_files:
            print(f"File '{existing_files[0]}' already exists. Skipping download.")
            return

    url = f"https://www.star.nesdis.noaa.gov/smcd/emb/vci/VH/get_TS_admin.php?country=UKR&provinceID={province_id}&year1={year1}&year2={year2}&type=Mean"
    response = urllib.request.urlopen(url)
    current_datetime = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')
    filename = f'vhi_id__{province_id}__{current_datetime}.csv'
    file_path = os.path.join(data_folder, filename)
    with open(file_path, 'wb') as f:
        f.write(response.read())
    print(f"Downloaded and saved to: {file_path}")

# === STEP 2: Create single DataFrame from all CSV files ===
def create_data_frame(folder_path):
    csv_files = glob.glob(folder_path + "/*.csv")
    headers = ['Year', 'Week', 'SMN', 'SMT', 'VCI', 'TCI', 'VHI', 'empty']
    frames = []

    for file in csv_files:
        region_id = int(file.split('__')[1])
        df = pd.read_csv(file, header=1, names=headers)
        df.at[0, 'Year'] = df.at[0, 'Year'][9:]
        df = df.drop(df.index[-1])
        df = df[df['VHI'] != -1]
        df = df.drop('empty', axis=1)
        df.insert(0, 'region_id', region_id)
        df['Week'] = df['Week'].astype(int)
        frames.append(df)

    result = pd.concat(frames).drop_duplicates().reset_index(drop=True)
    result = result.loc[(result.region_id != 12) & (result.region_id != 20)]
    result = result.replace({'region_id':{1:22, 2:24, 3:23, 4:25, 5:3, 6:4, 7:8, 8:19, 9:20, 10:21,
                                          11:9, 13:10, 14:11, 15:12, 16:13, 17:14, 18:15, 19:16, 21:17, 
                                          22:18, 23:6, 24:1, 25:2, 26:6, 27:5}})
    return result

# === STEP 3: Load or download data ===
# for i in range(1, 28):  # Uncomment to download all
#     download_data(i)

df = create_data_frame('./data')

# === STEP 4: Create region name dictionary ===
reg_id_name = {
    1: 'Вінницька',  2: 'Волинська',  3: 'Дніпропетровська',  4: 'Донецька',  5: 'Житомирська',
    6: 'Закарпатська',  7: 'Запорізька',  8: 'Івано-Франківська',  9: 'Київська',  10: 'Кіровоградська',
    11: 'Луганська',  12: 'Львівська',  13: 'Миколаївська',  14: 'Одеська',  15: 'Полтавська',
    16: 'Рівенська',  17: 'Сумська',  18: 'Тернопільська',  19: 'Харківська',  20: 'Херсонська',
    21: 'Хмельницька',  22: 'Черкаська',  23: 'Чернівецька',  24: 'Чернігівська',  25: 'Республіка Крим'
}

# === STEP 5: Interactive widgets for filtering and plotting ===
def update_plot(region_id, parameter, year_start, year_end, week_start, week_end):
    filtered = df[
        (df['region_id'] == region_id) &
        (df['Year'].astype(int) >= year_start) & (df['Year'].astype(int) <= year_end) &
        (df['Week'] >= week_start) & (df['Week'] <= week_end)
    ]

    pivot_data = filtered.pivot(index='Year', columns='Week', values=parameter)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    sns.heatmap(pivot_data, cmap="inferno", annot=False, ax=ax1)
    ax1.set_title(f"Heatmap {parameter} for {reg_id_name.get(region_id, 'Unknown')}")

    for year in filtered['Year'].unique():
        subset = filtered[filtered['Year'] == year]
        sns.lineplot(data=subset, x='Week', y=parameter, ax=ax2, label=str(year))

    ax2.set_title(f"{parameter} Trends by Week for {reg_id_name.get(region_id, 'Unknown')}")
    ax2.legend(loc='best')
    plt.tight_layout()
    plt.show()

interact(
    update_plot,
    region_id=widgets.Dropdown(options=sorted(df['region_id'].unique()), description='Регіон'),
    parameter=widgets.Dropdown(options=['VCI', 'TCI', 'VHI'], description='Параметр'),
    year_start=widgets.IntSlider(min=1981, max=2024, step=1, value=2000, description='Рік від'),
    year_end=widgets.IntSlider(min=1981, max=2024, step=1, value=2024, description='Рік до'),
    week_start=widgets.IntSlider(min=1, max=52, step=1, value=1, description='Тиждень від'),
    week_end=widgets.IntSlider(min=1, max=52, step=1, value=10, description='Тиждень до')
)
