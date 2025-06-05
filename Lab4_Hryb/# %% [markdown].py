# %% [markdown]
# ФБ-31 Гриб Вероніка
# 
# Лабораторна робота №4
# 
# Структури для роботи з великими обсягами даних в Python
# 
# Мета: отримати навички роботи із структурами для зберігання в Python (python, numpy, pandas, numpy array, dataframe, timeit) Основні поняття: numpy масиви, кортежі, списки, фрейми, профілювання.
# 
# Теоретичні відомості
# 
# Мінімально необхідні навички роботи із масивами та фреймами вже отримано при виконанні лабораторних робіт 1 та 2. За потреби можна скористатись офіційними сторінками відповідних проектів:
# 
# http://pandas.pydata.org/pandas-docs/version/0.15.2/index.html
# 
# https://docs.scipy.org/doc/scipy/

# %%
import pandas as pd
import numpy as np
import time
from tabulate import tabulate
from datetime import time as dt_time

print("Setup ok")

# %%
# Завантаження та підготовка даних

dtype = {
    "Global_active_power": "float32",
    "Global_reactive_power": "float32",
    "Voltage": "float32",
    "Global_intensity": "float32",
    "Sub_metering_1": "float32",
    "Sub_metering_2": "float32",
    "Sub_metering_3": "float32",
}
na_values = ["?", "NA", "nan", ""]

df = pd.read_csv(
    "household_power_consumption.csv",
    sep=";",                
    dtype=dtype,
    na_values=na_values,
    parse_dates={"DateTime": ["Date", "Time"]}, 
    infer_datetime_format=True,
)

# Прибраю можливі пропуски
df.dropna(inplace=True)

# За можливої потреби розбиваю знову на Date і Time
df["Date"] = df["DateTime"].dt.date
df["Time"] = df["DateTime"].dt.time

np_arr = df.to_numpy()

print("DataFrame:")
print(df.head())          # або власна функція print_head

print("\nNumPy масив:")
print(np_arr[:5])         # замість print_head(np_arr)

# %%
def print_head(data, n=5):
    if isinstance(data, pd.DataFrame):
        print(tabulate(data.head(n), headers='keys', tablefmt='fancy_grid'))
    elif isinstance(data, np.ndarray):
        print(tabulate(data[:n], tablefmt='fancy_grid'))

# %%
#Завдання 1: коли активна потужність буде більшою за 5 кВт
def high_power_filter(df):
    return df.query("Global_active_power > 5")

def high_power_filter_np(arr):
    return arr[arr[:, 2].astype(float) > 5]

print("\nЗавдання 1:")
start_pd = time.time()
result_pd = high_power_filter(df)
print_head(result_pd)
duration_pd = time.time() - start_pd

start_np = time.time()
result_np = high_power_filter_np(np_arr)
print_head(result_np)
duration_np = time.time() - start_np

print(f"Час виконання:\nPd: {duration_pd} Np:{duration_np} сек")

# %%
#Завдання 2: коли напруга буде вище 235 В

def voltage_over_threshold(df, threshold=235):
    return df[df['Voltage'] > threshold]

def voltage_over_threshold_np(arr, threshold=235):
    return arr[arr[:, 4].astype(float) > threshold]

print("\nЗавдання 2:")
start_pd = time.time()
result_pd = voltage_over_threshold(df)
print_head(result_pd)
duration_pd = time.time() - start_pd

start_np = time.time()
result_np = voltage_over_threshold_np(np_arr)
print_head(result_np)
duration_np = time.time() - start_np

print(f"Час виконання:\nPd: {duration_pd} Np:{duration_np} сек")

# %%
#Завдання 3: коли сила струму в проміжку 19-20А та порівняння споживання
# Функція для фільтрації з DataFrame
def current_range_filter(df):
    cond = (df['Global_intensity'].between(19, 20)) & (df['Sub_metering_2'] > df['Sub_metering_3'])
    return df[cond]

# Функція для фільтрації з NumPy
def current_range_filter_np(np_arr, df):
    idx_intensity = df.columns.get_loc('Global_intensity')
    idx_sm2 = df.columns.get_loc('Sub_metering_2')
    idx_sm3 = df.columns.get_loc('Sub_metering_3')

    cond = (
        (np_arr[:, idx_intensity] >= 19) &
        (np_arr[:, idx_intensity] <= 20) &
        (np_arr[:, idx_sm2] > np_arr[:, idx_sm3])
    )

    return np_arr[cond]

print("\nЗадача 3")

# Обробка DataFrame
start_time = time.time()
filtered_df = current_range_filter(df)
print_head(filtered_df)
end_time = time.time() - start_time

# Обробка NumPy-масиву (передаємо також df)
start_time_np = time.time()
filtered_np = current_range_filter_np(np_arr, df)
print_head(filtered_np)
end_time_np = time.time() - start_time_np

print(f"Час виконання:\nPandas: {end_time:.6f} сек\nNumPy: {end_time_np:.6f} сек")

# %%
#Завдання 4: випадкова вибірка 500000 записів
def random_sample_stats(df, sample_size=500_000, seed=1):
    """Середнє Sub_metering_1/2/3 для випадкової вибірки у Pandas."""
    n = min(sample_size, len(df))             # аби не вибрати більше, ніж є
    sample_df = df.sample(n=n, replace=False, random_state=seed)
    means = sample_df[['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3']].mean()
    return means

def random_sample_stats_np(np_arr, df, sample_size=500_000, seed=1):
    """Те саме для NumPy-масиву (потребує df, щоб знайти індекси колонок)."""
    # 1. Індекси потрібних колонок
    idx_sm1 = df.columns.get_loc('Sub_metering_1')
    idx_sm2 = df.columns.get_loc('Sub_metering_2')
    idx_sm3 = df.columns.get_loc('Sub_metering_3')

    # 2. Розмір вибірки
    n_rows = np_arr.shape[0]
    n = min(sample_size, n_rows)

    # 3. Випадкові індекси з фіксованим seed
    rng = np.random.default_rng(seed)
    indices = rng.choice(n_rows, n, replace=False)

    # 4. Формуємо вибірку та рахуємо середнє
    sample = np_arr[indices][:, [idx_sm1, idx_sm2, idx_sm3]].astype(float)
    means = sample.mean(axis=0)   # повертається np.ndarray довжини 3
    return means

print("\nЗадача 4")

# Pandas
start = time.time()
means_df = random_sample_stats(df)
print("Pandas mean values:")
print(means_df)
elapsed_df = time.time() - start

# NumPy
start = time.time()
means_np = random_sample_stats_np(np_arr, df)
print("\nNumPy mean values:")
print(dict(zip(['Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3'], means_np)))
elapsed_np = time.time() - start

print(f"\nЧас виконання:\nPandas: {elapsed_df:.6f} сек\nNumPy:  {elapsed_np:.6f} сек")

# %%
#Завдання 5: відбір після 18:00+ 
from datetime import time as dt_time

def evening_heavy_users(df):
    times = pd.to_datetime(df["Time"], format="%H:%M:%S").dt.time
    cond = (
        (times > dt_time(18, 0)) &
        (df["Global_intensity"] > 6) &
        (df["Sub_metering_2"] > df["Sub_metering_1"]) &
        (df["Sub_metering_2"] > df["Sub_metering_3"])
    )
    return df[cond]

def evening_heavy_users_np(np_arr, df):
    # індекси потрібних колонок
    idx_time      = df.columns.get_loc("Time")
    idx_intensity = df.columns.get_loc("Global_intensity")
    idx_sm1       = df.columns.get_loc("Sub_metering_1")
    idx_sm2       = df.columns.get_loc("Sub_metering_2")
    idx_sm3       = df.columns.get_loc("Sub_metering_3")

    # масив із часом
    times = np_arr[:, idx_time]

    # якщо час збережено як рядки
    if isinstance(times[0], str):
        times = np.array(
            [dt_time(*map(int, t.split(":"))) for t in times],
            dtype=object                # обов’язково object, інакше NumPy «сплющить» усе у float
        )
    # якщо час збережено як datetime64 – конвертуємо до .time()
    elif not isinstance(times[0], dt_time):
        times = np.array([pd.to_datetime(t).time() for t in times], dtype=object)

    cond = (
        (times > dt_time(18, 0)) &
        (np_arr[:, idx_intensity].astype(float) > 6) &
        (np_arr[:, idx_sm2].astype(float) > np_arr[:, idx_sm1].astype(float)) &
        (np_arr[:, idx_sm2].astype(float) > np_arr[:, idx_sm3].astype(float))
    )
    return np_arr[cond]


print("\nЗадача 5")

# Pandas
start_time = time.time()
filtered_df = evening_heavy_users(df)
print_head(filtered_df)
elapsed_df = time.time() - start_time

# NumPy
start_time_np = time.time()
filtered_np = evening_heavy_users_np(np_arr, df)
print_head(filtered_np)
elapsed_np = time.time() - start_time_np

print(f"Час виконання:\nPandas: {elapsed_df:.6f} сек\nNumPy:  {elapsed_np:.6f} сек")


