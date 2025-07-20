import pandas as pd
import numpy as np
import os


def count_digits_in_string(input_string):
  digit_count = 0
  for char in input_string:
      if char.isdigit():
          digit_count += 1
  return digit_count

def is_rise_fall_question(input_string):
  amount_digit = count_digits_in_string(input_string)
  if(amount_digit > 200):
    return True
  return False

def get_stock_df(test_df):
  test_df['type'] = test_df['query'].apply(is_rise_fall_question)
  stock_df = test_df[test_df['type']]
  return stock_df

def get_numeric_count_list(text_list):
  count = 0
  count_list = []
  for x in range(len(text_list)):
    for n in range(len(text_list[x])):
      if text_list[x][n].isnumeric() == True:
        count = count+1
    if len(text_list[x]) == 0:
      count_list.append(0)
    else:
      count_list.append(round(count/len(text_list[x]),1))
    count = 0
  return count_list


def change_dtype_float(input_df):
  date_df = input_df.iloc[:,0]
  input_df = input_df.iloc[:,1:]
  input_df = input_df.apply(pd.to_numeric, errors='coerce').astype(np.float32)
  # input_df.info()

  output_df = input_df
  output_df['date'] = date_df
  return output_df


def get_csv(text):
  '''
  input -> str
  output -> dataframe

  '''
  text_list = text.replace(':',' ')
  text_list = text.replace('\n\n','\n').split('\n')
  text_list = [text for text in text_list if len(text)>20]

  n_list = []
  # print(get_numeric_count_list(text_list))
  for n, x in enumerate(get_numeric_count_list(text_list)):
    if (x <= max(get_numeric_count_list(text_list))+0.1) and (x >= max(get_numeric_count_list(text_list))-0.1):
      n_list.append(n)

  # print(n_list)
  target_list = text_list[n_list[0]-1:n_list[-1]+1]

  target_list_clean = []
  for n in target_list:
    target_list_clean.append(n.strip().split(','))

  # target_list_clean[0] = [col.replace("%","").replace(" ","").replace("-","") for col in target_list_clean[0]]


  target_list_clean[0] = ['Context:date', 'open', 'high', 'low', 'close', 'adjclose', 'inc5', 'inc10', 'inc15', 'inc20', 'inc25', 'inc30']
  # print(target_list_clean[0])
  df = pd.DataFrame(index=list(range(len(target_list_clean[1:]))),columns=target_list_clean[0])

  data_list = target_list_clean[1:]

  for index, row in df.iterrows():
    for n, col in enumerate(data_list[index]):
      if(n>0):
        x = round(float(data_list[index][n]),3)
        # print(x)
        row.iloc[n] = x
      else:
        row.iloc[n] = data_list[index][n]

  df = change_dtype_float(df)
  return df


def cal_stat(df):
  # Load and preprocess
  df = get_csv(df)


  # Feature engineering
  df["volatility"] = df["high"] - df["low"]
  df["momentum"] = df["close"] - df["open"]
  df["trend_up"] = (df["close"] > df["open"]).astype(int)
  df["target_5d"] = (df["inc5"] > 0).astype(int)

  # Target distribution
  rise_count = df["target_5d"].sum()
  fall_count = len(df) - rise_count
  rise_pct = 100 * rise_count / len(df)
  
  # Correlation
  correlations = df.corr(numeric_only=True)
  target_corr = correlations["target_5d"].drop("target_5d").sort_values(ascending=False)

  # Grouped stats
  grouped_stats = df.groupby("target_5d")[["volatility", "momentum", "trend_up"]].mean().round(3)
  grouped_stats.index = grouped_stats.index.map({0: "Fall", 1: "Rise"})

  # Format correlation string
  corr_str = "\n".join([f"{k}: {v:.3f}" for k, v in target_corr.items()])

  # Format grouped feature stats
  group_stats_str = grouped_stats.to_string()

  # Generate LLM-ready prompt string
  prompt = f"""

  🔹 Dataset Summary:
  The dataset contains time series data with the following columns: open, high, low, close, adjusted close, and future price changes after 5, 10, 15, 20, 25, and 30 days (named inc-5, inc-10, ..., inc-30).

  🔹 Engineered Features:
  - volatility = high - low
  - momentum = close - open
  - trend_up = 1 if close > open else 0
  - target_5d = 1 if inc-5 > 0 else 0 (binary classification label)

  🔹 Target Distribution:
  - Rise: {rise_count} rows ({rise_pct:.2f}%)
  - Fall: {fall_count} rows ({100 - rise_pct:.2f}%)

  🔹 Correlation with target_5d (sorted):
  {corr_str}

  🔹 Feature Averages by Class:
  {group_stats_str}

  Interpretation:
  - Days labeled "Rise" tend to have higher momentum and are more likely to close above open (trend_up = 1).
  - "Fall" days tend to have near-zero or negative momentum and are less likely to trend upward.
  - Volatility is slightly higher on fall days, but not significantly.

  Given a new sample with features like momentum, trend_up, and volatility, determine whether it is more likely to result in a price rise or fall after 5 days based on these insights.
  """

  # Optional: save to file or print
  # print(prompt)
  # with open("eda_prompt.txt", "w") as f:
  #     f.write(prompt)
  return prompt