# -*- coding: utf-8 -*-

# Run this app with `python app.py` and
# visit http://127.0.0.1:8050/ in your web browser.

import dash
import dash_core_components as dcc
import dash_html_components as html
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV
from sklearn.linear_model import BayesianRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import neighbors
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegressionCV
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import scale
import mlbgame
from statsmodels.tsa.arima_model import ARIMA
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options

teams = pd.read_csv('Teams.csv')

# Drop unnecessary columns
drop_cols = ['lgID', 'divID', 'franchID', 'Rank', 'Ghome',
            'L', 'DivWin', 'WCWin', 'LgWin', 'WSWin',
            'SF', 'name', 'park', 'attendance', 'BPF', 'PPF',
            'teamIDBR', 'teamIDlahman45', 'teamIDretro', 'franchID']

df = teams.drop(drop_cols, axis = 1)

# Eliminating columns with null values
df = df.drop(['CS', 'HBP'], axis = 1)

#Filling null values
df['SO'] = df['SO'].fillna(df['SO'].median())
df['DP'] = df['DP'].fillna(df['DP'].median())
df['BB'] = df['BB'].fillna(df['BB'].median())
df['SB'] = df['SB'].fillna(df['SB'].median())

fig1 = px.histogram(df, x = 'W')
fig1.update_layout(
    title="Distribution of Wins",
    xaxis_title = "Wins",
    yaxis_title="",
)

def assign_win_bins(W):
    '''
    Creates bins for win column
    '''
    if W < 50: return 1
    if W >= 50 and W <= 69: return 2
    if W >= 70 and W <= 89: return 3
    if W >= 90 and W <= 109: return 4
    if W >= 100: return 5

df['win_bins'] = df['W'].apply(assign_win_bins)

# Scatter Plot of Wins vs. Year
fig2 = px.scatter(df, x = 'yearID', y = 'W', color = 'win_bins')
fig2.update_layout(
    title="Wins Scatter Plot",
    xaxis_title = "Year",
    yaxis_title="Wins",
)

# Remove rows pre-1900
df = df[df['yearID'] >= 1900]

df['R_per_game'] = df['R'] / df['G']

#Average Runs per Game DataFrame for Graphing
rpg = df.groupby(['yearID']).R_per_game.mean()

# Create line plot of MLB runs per Game vs. Year
fig3 = px.line(rpg)
fig3.update_layout(
    title="MLB Yearly Runs Per Game",
    xaxis_title = "Year",
    yaxis_title="MLB Runs per Game",
    showlegend = False
)

def assign_label(year):
    '''
    Creates "year_label" column, which gives information about
    # how certain years are related
    '''
    if year < 1920: return 1
    elif year >= 1920 and year <= 1941: return 2
    elif year >= 1942 and year <= 1945: return 3
    elif year >= 1946 and year <= 1962: return 4
    elif year >= 1963 and year <= 1976: return 5
    elif year >= 1977 and year <= 1992: return 6
    elif year >= 1193: return 7

# Add "label_year" column to "df"
df['year_label'] = df['yearID'].apply(assign_label)
df_dummy = pd.get_dummies(df['year_label'], prefix = 'era')
# Concatenate "df" and "df_dummy"
df = pd.concat([df, df_dummy], axis = 1)

def assign_decade(year):
    '''
    Convert years into decades and creates dummy variables
    '''
    if year < 1920: return 1910
    elif year >= 1920 and year <= 1929: return 1920
    elif year >= 1930 and year <= 1939: return 1930
    elif year >= 1940 and year <= 1949: return 1940
    elif year >= 1950 and year <= 1959: return 1950
    elif year >= 1960 and year <= 1969: return 1960
    elif year >= 1970 and year <= 1979: return 1970
    elif year >= 1980 and year <= 1989: return 1980
    elif year >= 1990: return 1990

df['decade_label'] = df['yearID'].apply(assign_decade)

df_decade = pd.get_dummies(df['decade_label'], prefix = 'decade')
df = pd.concat([df, df_decade], axis = 1)

# Drop unecessary columns
df = df.drop(['year_label', 'decade_label'], axis =  1)

#Create new feature for Runs Allowed Per Game
df['RA_per_game'] = df['RA'] / df['G']

#Create scatter plots for Runs per game vs. wins
fig4 = px.scatter(df, x = 'R_per_game', y = 'W')
fig4.update_layout(
    title="Runs Per Game vs. Wins",
    xaxis_title = "Runs per Game",
    yaxis_title="Wins",
)

#Create scatter plots for Runs Allowed per game vs. wins
fig5 = px.scatter(df, x = 'RA_per_game', y = 'W')
fig5.update_layout(
    title="Runs Allowed Per Game vs. Wins",
    xaxis_title = "Runs Allowed per Game",
    yaxis_title="Wins",
)

#Create 3-D scatter plot for Wins vs. Runs per Game and Runs Allowed Per Game
fig6 = px.scatter_3d(df, x = 'R_per_game', y = 'RA_per_game', z = 'W')
fig6.update_layout(
    title="Wins vs. Runs Per Game and Runs Allowed Per Game",
)

attributes = ['G', 'R', 'AB', 'H', '2B', '3B', 'HR', 'BB', 'SO',
             'SB', 'RA', 'ER', 'ERA', 'CG', 'SHO', 'SV', 'IPouts',
             'HA', 'HRA', 'BBA', 'SOA', 'E', 'DP', 'FP',
             'era_1', 'era_2', 'era_3', 'era_4', 'era_5', 'era_6', 'era_7',
             'decade_1910', 'decade_1920', 'decade_1930', 'decade_1940',
             'decade_1950', 'decade_1960', 'decade_1970', 'decade_1980',
             'decade_1990', 'R_per_game', 'RA_per_game']

data_attributes = df[attributes]

expos = df[df['teamID'] == 'MON'].reset_index(drop = True)
expos_wins = expos.groupby(['yearID']).W.mean()

#Create line plot of Expos wins per year
fig7 = px.line(expos_wins)
fig7.update_layout(
    title="Expos Wins Per Year",
    xaxis_title = "Year",
    yaxis_title="Expos Wins",
    showlegend = False
)

#Create silhouette score directory
s_score_dict = {}
for i in range(2,11):
    km = KMeans(n_clusters = i, random_state = 42)
    l = km.fit_predict(data_attributes)
    s_s = metrics.silhouette_score(data_attributes, l)
    s_score_dict[i] = s_s

#Create K-means model and determine euclidean distances for each data point
kmeans_model = KMeans(n_clusters = 6, random_state = 42)
distances = kmeans_model.fit_transform(data_attributes)

labels = kmeans_model.labels_

#Create scatter plot using labels from K-means model as colour
fig8 = px.scatter(distances[:,0], distances[:,1], color = labels)
fig8.update_layout(
    title="KMeans Clusters",
    xaxis_title = "",
    yaxis_title="",
)

#Filter out seasons that are less than or equal to 150 games
df_long = df[df.G > 150]

data_attributes = df_long[attributes]

#Create K-means model and determine euclidean distances for each data point
kmeans_model = KMeans(n_clusters = 6, random_state = 42)
distances = kmeans_model.fit_transform(data_attributes)

labels_long = kmeans_model.labels_

#Create scatter plot using labels from K-means model as colour
fig9 = px.scatter(distances[:,0], distances[:,1], color = labels_long)
fig9.update_layout(
    title="KMeans Clusters (Seasons Longer Than 150 Games)",
    xaxis_title = "",
    yaxis_title="",
)

#Add labels from K-means clustering to DataFrame and attributes list
df['labels'] = labels
attributes.append('labels')
df.head(3)

#Split the DataFrame into train and test sets
X = df[attributes]
y = df['W']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

attributes.remove('G')
attributes.remove('R')
attributes.remove('AB')
attributes.remove('H')
attributes.remove('HR')
attributes.remove('2B')
attributes.remove('3B')
attributes.remove('SO')
attributes.remove('SB')
attributes.remove('RA')
attributes.remove('BB')
attributes.remove('ER')
attributes.remove('ERA')
attributes.remove('CG')
attributes.remove('IPouts')
attributes.remove('HA')
attributes.remove('HRA')
attributes.remove('BBA')
attributes.remove('SHO')
attributes.remove('SV')
attributes.remove('SOA')
attributes.remove('E')
attributes.remove('DP')
attributes.remove('FP')
attributes.remove('era_1')
attributes.remove('era_2')
attributes.remove('era_3')
attributes.remove('era_4')
attributes.remove('era_5')
attributes.remove('era_6')
attributes.remove('era_7')
attributes.remove('decade_1910')
attributes.remove('decade_1920')
attributes.remove('decade_1930')
attributes.remove('decade_1940')
attributes.remove('decade_1950')
attributes.remove('decade_1960')
attributes.remove('decade_1970')
attributes.remove('decade_1980')
attributes.remove('decade_1990')
attributes.remove('labels')

#Split the DataFrame into train and test sets
X = df[attributes]
y = df['W']

#Preprocessing
#X = scale(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Create Linear Regression model, fit model and make predictions
lr = LinearRegression()
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)

#Determine mean absolute error
mae_lr = mean_absolute_error(y_test, preds_lr)

est = sm.OLS(y_train, X_train).fit(cov_type = 'HC2')
preds = est.predict(X_test)

#Create Ridge Linear Regression model, fit model, and make predictions
rrm = RidgeCV(alphas = (0.01, 0.1, 1.0, 10.0), normalize = True)
rrm.fit(X_train, y_train)
preds_rrm = rrm.predict(X_test)

#Determine mean absolute error
mae_rrm = mean_absolute_error(y_test, preds_rrm)

#Create Naive Bayes model, fit model, and make predictions
nb = BayesianRidge(compute_score = True)
nb.fit(X_train, y_train)
preds_nb = nb.predict(X_test)

#Determine mean absolute error
mae_nb = mean_absolute_error(y_test, preds_nb)
mae_nb

#Create Decision Tree model, fit model, and make predictions
tm = DecisionTreeRegressor()
tm.fit(X_train, y_train)
preds_tm = tm.predict(X_test)

#Determine mean absolute error
mae_tm = mean_absolute_error(y_test, preds_tm)

#Create Random Forest model, fit model, and make predictions
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)

#Determine mean absolute error
mae_rf = mean_absolute_error(y_test, preds_rf)

#Create K-Nearest Neighbor model, fit model, and make predictions
knn = neighbors.KNeighborsRegressor(n_neighbors = 10)
knn.fit(X_train, y_train)
preds_knn = knn.predict(X_test)

#Determine mean absolute error
mae_knn = mean_absolute_error(y_test, preds_knn)

#Create Support Vector Classifier model, fit model, and make predictions
sv = SVC()
sv.fit(X_train, y_train)
preds_sv = sv.predict(X_test)

#Determine mean absolute error
mae_sv = mean_absolute_error(y_test, preds_sv)

#Create Stochastic Gradient Descent model, fit model, and make predictions
sgd = SGDClassifier(random_state = 42)
sgd.fit(X_train, y_train)
preds_sgd = sgd.predict(X_test)

#Determine mean absolute error
mae_sgd = mean_absolute_error(y_test, preds_sgd)

#Create Gaussian Naive Bayes model, fit model, and make predictions
gnb = GaussianNB()
gnb.fit(X_train, y_train)
preds_gnb = gnb.predict(X_test)

#Determine mean absolute error
mae_gnb = mean_absolute_error(y_test, preds_gnb)

#Add squared terms to Linear Regression Model
for a in attributes:
    X[a + '2'] = X.R_per_game ** 2

#Split the DataFrame into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

#Create Linear Regression model, fit model and make predictions
lr2 = LinearRegression()
lr2.fit(X_train, y_train)
preds_lr2 = lr2.predict(X_test)

#Determine mean absolute error
mae_lr2 = mean_absolute_error(y_test, preds_lr2)

models = ['Linear Regressions with Squared Terms',
          'Linear Regression',
          'Naive Bayes',
          'Ridge Linear Regression',
          'K-Nearest Neighbor',
          'Support Vector Classifier',
          'Random Forest',
          'Gaussian Naive Bayes',
          'Decision Tree',
          'Stochastic Gradient Descent'
          ]
mae = [mae_lr2, mae_lr, mae_nb, mae_rrm, mae_knn, mae_sv, mae_rf, mae_gnb, mae_tm, mae_sgd]
df_models = pd.DataFrame(mae, models)
df_models.columns = ['Mean Absolute Error']

fig10 = px.bar(df_models, y= 'Mean Absolute Error', color = 'Mean Absolute Error')
fig10.update_layout(
    title="Model Comparison",
    xaxis_title = "Model",
)

#Filter for 1994 NL East Teams
NL_East = df[(df['yearID'] == 1994)
            & ((df['teamID'] == 'ATL')
             | (df['teamID'] == 'FLO')
             | (df['teamID'] == 'MON')
             | (df['teamID'] == 'NYN')
             | (df['teamID'] == 'PHI'))
            ].reset_index(drop = True)

X = NL_East[attributes]

for a in attributes:
    X[a + '2'] = X.R_per_game ** 2

preds_lr2 = lr2.predict(X)

NL_East['pred_wins'] = np.round(preds_lr2)

NL_East['mae'] = mae_lr2

NL_East = NL_East.sort_values(by=['pred_wins'], ascending = False)

fig11 = px.bar(NL_East, y= 'pred_wins', x = 'teamID', color = 'pred_wins', error_y = 'mae')
fig11.update_layout(
    title="NL East Predicted Wins",
    xaxis_title = "Team",
    yaxis_title = "Predicted Wins",
)

def mlb_wins(team, month):
    '''
    Given the team, and the month, return the number of wins in 1994 season.
    '''
    wins = 0
    YEAR = 1994
    
    #Determine number of home wins
    home = mlbgame.games(YEAR, month, home = team)
    home_games = mlbgame.combine_games(home)

    for game in home_games:
        if game.w_team == team:
            wins += 1
            
    #Determine number of home wins
    away = mlbgame.games(YEAR, month, away = team)
    away_games = mlbgame.combine_games(away)

    for game in away_games:
        if game.w_team == team:
            wins += 1
        
    return wins, len(home_games + away_games)

months = [4, 5, 6, 7, 8]

Expos = {}

for m in months:
    Expos[m] = mlb_wins('Expos', m)

df_Expos = pd.DataFrame.from_dict(Expos, orient = 'index')

df_Expos.index.name = 'month'
df_Expos.columns = ['wins', 'games']

def win_percentage(df):
    '''
    Returns the win percentage time history of a given dataframe
    '''
    wins = 0
    games = 0
    win_percent = 0
    total_win_percentage = []

    for i, r in df.iterrows():
        wins += r['wins']
        games += r['games']
        win_percent = wins/games
        total_win_percentage.append(win_percent)
    
    return total_win_percentage

df_Expos['win_percentage'] = win_percentage(df_Expos)

df_Expos.index = [pd.datetime(1994, month+1, 1) for month in df_Expos.index]

df_Expos_wp = df_Expos.win_percentage

model = ARIMA(df_Expos_wp, order = (1,0,0))
res = model.fit()
Expos_preds = res.predict('1994-05-01', '1994-10-01')

df_Expos_preds = pd.concat([df_Expos_wp, Expos_preds], axis = 1)
df_Expos_preds.columns = ['win_percentage', 'forecast']

fig12 = px.line(df_Expos_preds)
fig12.update_layout(
    title="Expos Forecasted Win Percentage",
    xaxis_title = "Time",
    yaxis_title="Win Percentage",
)

Braves = {}

for m in months:
    Braves[m] = mlb_wins('Braves', m)

df_Braves = pd.DataFrame.from_dict(Braves, orient = 'index')

df_Braves.index.name = 'month'
df_Braves.columns = ['wins', 'games']

df_Braves['win_percentage'] = win_percentage(df_Braves)

df_Braves.index = [pd.datetime(1994, month+1, 1) for month in df_Braves.index]

df_Braves_wp = df_Braves.win_percentage

model = ARIMA(df_Braves_wp, order = (1,0,0))
res = model.fit()
Braves_preds = res.predict('1994-05-01', '1994-10-01')

df_Braves_preds = pd.concat([df_Braves_wp, Braves_preds], axis = 1)
df_Braves_preds.columns = ['win_percentage', 'forecast']

fig13 = px.line(df_Braves_preds)
fig13.update_layout(
    title="Braves Forecasted Win Percentage",
    xaxis_title = "Time",
    yaxis_title="Win Percentage",
)

app.layout = html.Div(children=[
    html.H1(children='MLB 1994 NL East Pennant Prediction'),

    html.H2(children='Introduction'),

    html.Div(children='''
        In 1994, everything was finally coming together for the Montreal Expos.  They had won 74 of their first 114 games.
        Then on August 11th, a player's strike cancelled the remainder of the season, and the World Series were not played
        for the first time in 90 years.

        We will attempt to predict how the 1994 season might have played out were it not for this strike using data from
        the Baseball Database at SeanLahman.com.
    '''),

    html.H2(children='Data Visualization'),

        html.Div(children='''
        We start with a histogram of the target column to see the distribution of wins.
    '''),

    dcc.Graph(
        figure=fig1
    ),

    html.Div(children='''
        Next, we create a scatter plot of the wins versus year.  The colours represent bins that have been created for the
        number of wins.
    '''),

    dcc.Graph(
        figure=fig2
    ),

        html.Div(children='''
        We then add a column for the number of runs per game, then create a line plot of the number of the average runs per game
        versus year.
    '''),

        dcc.Graph(
        figure=fig3
    ),

        html.Div(children='''
        We can also plot the number of wins versus the average runs per game.
    '''),

        dcc.Graph(
        figure=fig4
    ),

        html.Div(children='''
        Similarly, we can plot the number of wins versus the average runs allowed per game.
    '''),

        dcc.Graph(
        figure=fig5
    ),

        html.Div(children='''
        We can then combine these plots to show the number of wins versus the runs per game and runs allowed per game.
    '''),

        dcc.Graph(
        figure=fig6
    ),

        html.Div(children='''
        Next, we can filter by team name to get the number of Expos wins each year.
    '''),

        dcc.Graph(
        figure=fig7
    ),

        html.H2(children='K-Means Clustering'),

        html.Div(children='''
        The silhouette_score() function was used to determine a number of clusters of 6. We determine the Euclidean distances
        for each data point using the fit_transform() method, and visualize these clusters with a scatter plot.
    '''),

        dcc.Graph(
        figure=fig8
    ),

        html.Div(children='''
        The points with a Euclidean distances greater than 2500 are from seasons that were less than the usual 162 games, such as
        the 1994 season.  If we temporarily filter our dataframe to remove seasons that are less than or equal to 130 games, we get
        the following scatter plot.
    '''),

        dcc.Graph(
        figure=fig9
    ),

        html.H2(children='Modeling'),

        html.Div(children='''
        First, we split the data into training and test dataframes.  80% of the data was used to train the model.  Like most team
        sports, the primary goal of baseball is to outscore your opponent.  The attributes list was greatly simplified from 
        the original 43 attributes to include only the average runs scored and runs allowed per game.
        10 models were used to predict the number of wins in a season, and the mean absolute errors of each are shown below.
    '''),

        dcc.Graph(
        figure=fig10
    ),

        html.H2(children='NL East Predictions'),

        html.Div(children='''
        We insert the average runs and runs allowed per game into our Linear Regression model with squared terms to play out 
        the entire season. The teams are sorted by the predicted number of wins.
    '''),

        dcc.Graph(
        figure=fig11
    ),

        html.H2(children='ARIMA Forecasting'),

        html.Div(children='''
        Using the mlbgame API, the Expos' winning percentage was determined for each month.  An ARIMA model was fitted to this data
        to forecast the Expos' winning percentage at the end of the season.
    '''),

        dcc.Graph(
        figure=fig12
    ),

        html.Div(children='''
        Based on the predictions made earlier, the only team that had a chance of catching the Expos were the Atlanta Braves.
        The Braves' winning perecentage at the end of the season is forecasted below.
    '''),

        dcc.Graph(
        figure=fig13
    ),

        html.H2(children='Conclusion'),

        html.Div(children='''
        After the data was visualized, 10 models were used to predict the number of wins in a season using only the average runs
        and the average runs allowed per game.  Of these, the linear regression model with squared terms was used, since it
        minized the mean absolute error.  This model predicted the Expos would win 96 games if the season would have been played
        in its entirety. If this were the case, this would have been the first and only Expos pennant win in their 35-year history.
        This prediction was then validated using ARIMA forecasting.
    ''')

])

if __name__ == '__main__':
    app.run_server(debug=True)