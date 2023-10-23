import os

from tqdm import tqdm
import pandas as pd


def get_cm_per_country(df, time_col, country, feature='temperature'):
    df = df[[time_col, country + '_' + feature]].copy()
    df['date'] = df.apply(lambda x: pd.to_datetime(x[time_col]).date(), axis=1)
    df['time'] = df.apply(lambda x: pd.to_datetime(x[time_col]).time(), axis=1)
    df = df[['date', 'time', country + '_' + feature]]
    df = df.pivot(index='date', columns='time')
    df.columns = df.columns.droplevel(0)
    df.columns = [feature + '_' + str(i) for i in df.columns]
    return df


def get_all_features_per_country(df, time_col, country):
    features = ['temperature', 'radiation_direct_horizontal',
                'radiation_diffuse_horizontal']
    result = pd.DataFrame()
    for feature in features:
        temp = get_cm_per_country(df, time_col, country, feature)
        result = pd.concat([result, temp], axis=1)

    hours = list(dict.fromkeys([col.rsplit('_', 1)[-1] for col in result.columns]))
    features = list(dict.fromkeys([col.rsplit('_', 1)[0] for col in result.columns]))
    for hour in hours:
        sel_cols = [feat + '_' + hour for feat in features]
        result[hour] = result.apply(lambda x: [x[sel_cols[0]], x[sel_cols[1]], x[sel_cols[2]]], axis=1)
    result = result[hours]
    return result


def get_trajectories_per_year(df, time_col, country, folder_path):
    print(f"INFO: Preprocessing trajectories for {country}...")
    years = [str(y) for y in pd.to_datetime(df[time_col]).apply(lambda x: x.year).unique()]
    for year in tqdm(years):
        mask = df[time_col].apply(lambda x: any(item for item in [year] if item in str(x)))
        temp = df[mask]
        temp = get_all_features_per_country(temp, time_col, country)
        temp.to_csv(folder_path + "/weather_" + country + '_' + year + '.csv', index=False)


def extract_country_to_file(country, folder_path):
    raw_data = pd.read_csv('data/weather_data.csv')
    os.makedirs(folder_path, exist_ok=True)
    print('INFO: Created directory ', folder_path)

    get_trajectories_per_year(raw_data, 'utc_timestamp', country, folder_path)