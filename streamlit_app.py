import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dateutil import relativedelta as datere
import matplotlib.pyplot as plt
import matplotlib
import streamlit as st
import time
from warnings import simplefilter

# Configure matplotlib font
matplotlib.font_manager.fontManager.addfont('TaipeiSansTCBeta-Regular.ttf')
matplotlib.rc('font', family='Taipei Sans TC Beta')
simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

# Streamlit page setup
st.set_page_config(page_title="Jockey Race 賽馬程式")
st.title("Jockey Race 賽馬程式")

# Global dictionaries
odds_dict = {}
investment_dict = {}
overall_investment_dict = {}
weird_dict = {}
diff_dict = {}
post_time_dict = {}
numbered_dict = {}

# Benchmark constants
BENCHMARKS = {
    "WIN": 10,
    "PLA": 100,
    "QIN": 50,
    "QPL": 100
}

# Function to fetch investment data
def get_investment_data(Date, place, race_no, methodlist):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "racing",
        "variables": {"date": str(Date), "venueCode": place, "raceNo": int(race_no), "oddsTypes": methodlist},
        "query": """
            query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
              raceMeetings(date: $date, venueCode: $venueCode) {
                poolInvs: pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
                  id
                  oddsType
                  investment
                }
              }
            }
        """
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        investments = {k: [] for k in ["WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "FF"]}
        race_meetings = data.get('data', {}).get('raceMeetings', [])
        for meeting in race_meetings:
            for pool in meeting.get('poolInvs', []):
                if place not in ['ST','HV']:
                    if pool.get('id', '')[8:10] != place:
                        continue
                odds_type = pool.get('oddsType')
                investment = float(pool.get('investment', 0))
                if odds_type in investments:
                    investments[odds_type].append(investment)
        return investments
    except Exception as e:
        st.error(f"Error fetching investment data: {e}")
        return None

# Function to fetch odds data
def get_odds_data(Date, place, race_no, methodlist):
    url = 'https://info.cld.hkjc.com/graphql/base/'
    headers = {'Content-Type': 'application/json'}
    payload = {
        "operationName": "racing",
        "variables": {"date": str(Date), "venueCode": place, "raceNo": int(race_no), "oddsTypes": methodlist},
        "query": """
            query racing($date: String, $venueCode: String, $oddsTypes: [OddsType], $raceNo: Int) {
              raceMeetings(date: $date, venueCode: $venueCode) {
                pmPools(oddsTypes: $oddsTypes, raceNo: $raceNo) {
                  id
                  oddsType
                  oddsNodes {
                    combString
                    oddsValue
                  }
                }
              }
            }
        """
    }
    try:
        response = requests.post(url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        odds_values = {k: [] for k in ["WIN", "PLA", "QIN", "QPL", "FCT", "TRI", "FF"]}
        race_meetings = data.get('data', {}).get('raceMeetings', [])
        for meeting in race_meetings:
            for pool in meeting.get('pmPools', []):
                if place not in ['ST','HV']:
                    if pool.get('id', '')[8:10] != place:
                        continue
                odds_type = pool.get('oddsType')
                if odds_type not in odds_values:
                    continue
                nodes = pool.get('oddsNodes', [])
                odds_values[odds_type] = []
                for node in nodes:
                    val = node.get('oddsValue')
                    if val == 'SCR':
                        val = np.inf
                    else:
                        try:
                            val = float(val)
                        except:
                            continue
                    if odds_type in ("QIN","QPL","FCT","TRI","FF"):
                        comb = node.get('combString')
                        if comb:
                            odds_values[odds_type].append((comb, val))
                    else:
                        odds_values[odds_type].append(val)
        for key in ["QIN","QPL","FCT","TRI","FF"]:
            odds_values[key].sort(key=lambda x: x[0])
        return odds_values
    except Exception as e:
        st.error(f"Error fetching odds data: {e}")
        return None

# Functions to save odds and investment data into dictionaries
def save_odds_data(time_now, odds, methodlist):
    for method in methodlist:
        if method in ['WIN', 'PLA']:
            if odds_dict.get(method) is None or odds_dict[method].empty:
                odds_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
            odds_dict[method].loc[time_now] = odds[method]
        else:
            if odds[method]:
                combs, vals = zip(*odds[method])
                if odds_dict.get(method) is None or odds_dict[method].empty:
                    odds_dict[method] = pd.DataFrame(columns=combs)
                odds_dict[method].loc[time_now] = vals

def save_investment_data(time_now, investment, odds, methodlist):
    for method in methodlist:
        if method in ['WIN', 'PLA']:
            if investment_dict.get(method) is None or investment_dict[method].empty:
                investment_dict[method] = pd.DataFrame(columns=np.arange(1, len(odds[method]) + 1))
            inv_vals = [round(investment[method][0] * 0.825 / 1000 / odd, 2) for odd in odds[method]]
            investment_dict[method].loc[time_now] = inv_vals
        else:
            if odds[method]:
                combs, vals = zip(*odds[method])
                if investment_dict.get(method) is None or investment_dict[method].empty:
                    investment_dict[method] = pd.DataFrame(columns=combs)
                inv_vals = [round(investment[method][0] * 0.825 / 1000 / v, 2) for v in vals]
                investment_dict[method].loc[time_now] = inv_vals

def investment_combined(time_now, method, df):
    sums = {}
    for col in df.columns:
        num1, num2 = map(int, col.split(','))
        col_sum = df[col].sum()
        sums[num1] = sums.get(num1, 0) + col_sum
        sums[num2] = sums.get(num2, 0) + col_sum
    return pd.DataFrame([sums], index=[time_now]) / 2

def get_overall_investment(time_now):
    no_of_horse = len(investment_dict['WIN'].columns)
    total_df = pd.DataFrame(index=[time_now], columns=np.arange(1, no_of_horse+1))
    for method in methodlist:
        if method in ['WIN', 'PLA']:
            overall_investment_dict[method] = overall_investment_dict[method]._append(investment_dict[method].tail(1))
        elif method in ['QIN', 'QPL']:
            overall_investment_dict[method] = overall_investment_dict[method]._append(investment_combined(time_now, method, investment_dict[method].tail(1)))
    for horse in range(1, no_of_horse+1):
        total = sum(overall_investment_dict[method][horse].values[-1] for method in methodlist)
        total_df[horse] = total
    overall_investment_dict['overall'] = overall_investment_dict['overall']._append(total_df)

def weird_data(investments, time_now):
    target_list = methodlist[:4]
    if 'QPL' not in target_list:
        target_list = methodlist[:3]
    for method in target_list:
        latest_investment = investment_dict[method].tail(1).values
        last_time_odds = odds_dict[method].tail(2).head(1)
        expected = investments[method][0] * 0.825 / 1000 / last_time_odds
        diff = round(latest_investment - expected, 0)
        if method in ['WIN', 'PLA']:
            diff_dict[method] = diff_dict[method]._append(diff)
        else:
            diff_dict[method] = diff_dict[method]._append(investment_combined(time_now, method, diff))
        diff.index = diff.index.strftime('%H:%M:%S')
        benchmark = BENCHMARKS.get(method, 10)
        for idx in investment_dict[method].tail(1).columns:
            error = diff[idx].values[0]
            if error > benchmark:
                if error < benchmark * 2:
                    highlight = '-'
                elif error < benchmark * 3:
                    highlight = '*'
                elif error < benchmark * 4:
                    highlight = '**'
                else:
                    highlight = '***'
                err_df = pd.DataFrame([[idx, error, odds_dict[method].tail(1)[idx].values, highlight]],
                                      columns=['No.', 'error', 'odds', 'Highlight'], index=diff.index)
                weird_dict[method] = weird_dict[method]._append(err_df)

def change_overall(time_now):
    if 'QPL' in methodlist[:4]:
        total = diff_dict['WIN'].sum(axis=0) + diff_dict['PLA'].sum(axis=0) + diff_dict['QIN'].sum(axis=0) + diff_dict['QPL'].sum(axis=0)
    else:
        total = diff_dict['WIN'].sum(axis=0) + diff_dict['PLA'].sum(axis=0) + diff_dict['QIN'].sum(axis=0)
    diff_dict['overall'] = diff_dict['overall']._append(pd.DataFrame([total], index=[time_now]))

def print_bar_chart(time_now):
    post_time = post_time_dict.get(race_no)
    if not post_time:
        return
    time_25m_before = np.datetime64(post_time - timedelta(minutes=25) + timedelta(hours=8))
    for method in print_list:
        df = overall_investment_dict.get(method)
        if df is None or df.empty or df.tail(1).sum(axis=1).values[0] == 0:
            continue
        change_data = diff_dict.get(method).tail(10).sum(axis=0) if method != 'overall' else diff_dict[method].iloc[-1]
        fig, ax = plt.subplots(figsize=(12, 6))
        df.index = pd.to_datetime(df.index)
        df_first = df[df.index < time_25m_before].tail(1)
        df_second = df[df.index >= time_25m_before].tail(1)
        change_df = pd.DataFrame([change_data.apply(lambda x: x*4 if x > 0 else x*2)], columns=change_data.index, index=[df.index[-1]])
        if df_first.empty:
            data_df = pd.DataFrame()
        else:
            data_df = df_first.append(df_second)
        sorted_data = data_df.sort_values(by=data_df.index[0], axis=1, ascending=False)
        X = sorted_data.columns
        X_axis = np.arange(len(X))
        sorted_change = change_df[X]
        bar_color = 'blue' if df_first.empty else 'red'
        if not df_first.empty:
            if df_second.empty:
                ax.bar(X_axis, sorted_data.iloc[0], 0.4, label='投注額', color='pink')
            else:
                ax.bar(X_axis - 0.2, sorted_data.iloc[1], 0.4, label='25分鐘', color=bar_color)
                ax.bar(X_axis + 0.2, sorted_change.iloc[0], 0.4, label='改變', color='grey')
        plt.xticks(X_axis, sorted_data.columns, fontsize=12)
        ax.grid(color='lightgrey', axis='y', linestyle='--')
        ax.set_ylabel('投注額', fontsize=15)
        fig.legend()
        plt.title({'overall':'綜合','QIN':'連贏','QPL':'位置Q','WIN':'獨贏','PLA':'位置'}.get(method, method), fontsize=15)
        st.pyplot(fig)

# Streamlit UI and logic
Date = st.date_input('日期:', value=datetime.now())
options = ['ST', 'HV', 'S1', 'S2', 'S3', 'S4', 'S5']
place = st.selectbox('場地:', options)
race_options = list(range(1, 12))
race_no = st.selectbox('場次:', race_options)

checkbox_no_qpl = st.checkbox('沒有位置Q', value=False)
if checkbox_no_qpl:
    methodlist = ['WIN', 'PLA', 'QIN', 'QPL', 'FCT', 'TRI', 'FF']
    methodCHlist = ['獨贏', '位置', '連贏', '位置Q', '二重彩', '單T', '四連環']
    print_list = ['qin_qpl', 'PLA', 'WIN']
else:
    methodlist = ['WIN', 'PLA', 'QIN', 'QPL', 'TRI']
    methodCHlist = ['獨贏', '位置', '連贏', '位置Q', '單T']
    print_list = ['QIN', 'QPL', 'PLA', 'WIN']

def click_start_button():
    st.session_state.reset = True

if 'reset' not in st.session_state:
    st.session_state.reset = False

if st.button('開始', on_click=click_start_button):
    # Initialize data structures for the run
    for method in methodlist:
        odds_dict[method] = pd.DataFrame()
        investment_dict[method] = pd.DataFrame()
        overall_investment_dict[method] = pd.DataFrame()
        weird_dict[method] = pd.DataFrame([], columns=['No.', 'error', 'odds', 'Highlight'])
        diff_dict[method] = pd.DataFrame()
    diff_dict['overall'] = pd.DataFrame()
    overall_investment_dict['overall'] = pd.DataFrame()

    # Fetch race meeting and runners info for postTime etc. (omitted here) - fill post_time_dict accordingly

    # Main loop for live data fetching and display
    start_time = time.time()
    end_time = start_time + 60 * 1000  # approximate 60,000 seconds for looping -- adjust as needed
    placeholder = st.empty()

    while time.time() <= end_time:
        with placeholder.container():
            time_now = datetime.now() + datere.relativedelta(hours=8)
            odds = get_odds_data(Date, place, race_no, methodlist)
            investments = get_investment_data(Date, place, race_no, methodlist)
            if odds is None or investments is None:
                st.warning("無法獲取賠率或投注數據。")
                break
            save_odds_data(time_now, odds, methodlist)
            save_investment_data(time_now, investments, odds, methodlist)
            get_overall_investment(time_now)
            weird_data(investments, time_now)
            change_overall(time_now)
            print_bar_chart(time_now)
            time.sleep(20)
