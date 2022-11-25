#Build the dashboard using Plotly dash

#Importing the Libraries
from dash import Dash, html, dcc  # importing the dash framework
from dash import dcc # importing the core components from dash
from dash import html # importing the html components from dash
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.graph_objs as go
from dash.dependencies import Input, Output
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# creating a new instance of Dash
app = Dash(__name__,external_stylesheets=[dbc.themes.UNITED])
server = app.server
#Data Normalization:Normalization is changing the values of numeric columns
#in the dataset to a common scale, which helps the performance of our model
scaler=MinMaxScaler(feature_range=(0,1))

#import NFLX dataSet for application
df_nse_NFLX = pd.read_csv("NFLX.csv")
df_nse_NFLX["Date"]=pd.to_datetime(df_nse_NFLX.Date,format="%Y-%m-%d")
df_nse_NFLX.index=df_nse_NFLX['Date']
data_NFLX=df_nse_NFLX.sort_index(ascending=True,axis=0)
new_data_NFLX=pd.DataFrame(index=range(0,len(df_nse_NFLX)),columns=['Date','Close'])

#import AAPL dataSet for application
df_nse_AAPL = pd.read_csv("AAPL.csv")
df_nse_AAPL["Date"]=pd.to_datetime(df_nse_AAPL.Date,format="%Y-%m-%d")
df_nse_AAPL.index=df_nse_AAPL['Date']
data_AAPL=df_nse_AAPL.sort_index(ascending=True,axis=0)
new_data_AAPL=pd.DataFrame(index=range(0,len(df_nse_AAPL)),columns=['Date','Close'])

#import META dataSet for application
df_nse_META = pd.read_csv("META.csv")
df_nse_META["Date"]=pd.to_datetime(df_nse_META.Date,format="%Y-%m-%d")
df_nse_META.index=df_nse_META['Date']
data_META=df_nse_META.sort_index(ascending=True,axis=0)
new_data_META=pd.DataFrame(index=range(0,len(df_nse_META)),columns=['Date','Close'])

#import MSFT dataSet for application
df_nse_MSFT = pd.read_csv("MSFT.csv")
df_nse_MSFT["Date"]=pd.to_datetime(df_nse_MSFT.Date,format="%Y-%m-%d")
df_nse_MSFT.index=df_nse_MSFT['Date']
data_MSFT=df_nse_MSFT.sort_index(ascending=True,axis=0)
new_data_MSFT=pd.DataFrame(index=range(0,len(df_nse_MSFT)),columns=['Date','Close'])

#import TSLA dataSet for application
df_nse_TSLA = pd.read_csv("TSLA.csv")
df_nse_TSLA["Date"]=pd.to_datetime(df_nse_TSLA.Date,format="%Y-%m-%d")
df_nse_TSLA.index=df_nse_AAPL['Date']
data_TSLA=df_nse_TSLA.sort_index(ascending=True,axis=0)
new_data_TSLA=pd.DataFrame(index=range(0,len(df_nse_TSLA)),columns=['Date','Close'])

# sort NFLX dataSet by Data, Close
for i in range(0,len(data_NFLX)):
    new_data_NFLX["Date"][i]=data_NFLX['Date'][i]
    new_data_NFLX["Close"][i]=data_NFLX["Close"][i]
new_data_NFLX.index=new_data_NFLX.Date
new_data_NFLX.drop("Date",axis=1,inplace=True)
dataset_NFLX=new_data_NFLX.values
train_NFLX=dataset_NFLX[0:1007,:]
valid_NFLX=dataset_NFLX[1007:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data_NFLX=scaler.fit_transform(dataset_NFLX)

# sort AAPL dataSet by Data, Close
for i in range(0,len(data_AAPL)):
    new_data_AAPL["Date"][i]=data_AAPL['Date'][i]
    new_data_AAPL["Close"][i]=data_AAPL["Close"][i]
new_data_AAPL.index=new_data_AAPL.Date
new_data_AAPL.drop("Date",axis=1,inplace=True)
dataset_AAPL=new_data_AAPL.values
train_AAPL=dataset_AAPL[0:1007,:]
valid_AAPL=dataset_AAPL[1007:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data_AAPL=scaler.fit_transform(dataset_AAPL)


# sort META dataSet by Data, Close
for i in range(0,len(data_META)):
    new_data_META["Date"][i]=data_META['Date'][i]
    new_data_META["Close"][i]=data_META["Close"][i]
new_data_META.index=new_data_META.Date
new_data_META.drop("Date",axis=1,inplace=True)
dataset_META=new_data_META.values
train_META=dataset_META[0:1007,:]
valid_META=dataset_META[1007:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data_META=scaler.fit_transform(dataset_META)


# sort MSFT dataSet by Data, Close
for i in range(0,len(data_MSFT)):
    new_data_MSFT["Date"][i]=data_MSFT['Date'][i]
    new_data_MSFT["Close"][i]=data_MSFT["Close"][i]
new_data_MSFT.index=new_data_MSFT.Date
new_data_MSFT.drop("Date",axis=1,inplace=True)
dataset_MSFT=new_data_MSFT.values
train_MSFT=dataset_MSFT[0:1007,:]
valid_MSFT=dataset_MSFT[1007:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data_MSFT=scaler.fit_transform(dataset_MSFT)


# sort TSLA dataSet by Data, Close
for i in range(0,len(data_TSLA)):
    new_data_TSLA["Date"][i]=data_TSLA['Date'][i]
    new_data_TSLA["Close"][i]=data_TSLA["Close"][i]
new_data_TSLA.index=new_data_TSLA.Date
new_data_TSLA.drop("Date",axis=1,inplace=True)
dataset_TSLA=new_data_TSLA.values
train_TSLA=dataset_TSLA[0:1007,:]
valid_TSLA=dataset_TSLA[1007:,:]
scaler=MinMaxScaler(feature_range=(0,1))
scaled_data_TSLA=scaler.fit_transform(dataset_TSLA)


#split the NFLX_data into x_train and y_train dataset
#Incorporating Timesteps Into Data
x_train_NFLX,y_train_NFLX=[],[]
for i in range(60,len(train_NFLX)):
    x_train_NFLX.append(scaled_data_NFLX[i-60:i,0])
    y_train_NFLX.append(scaled_data_NFLX[i,0])
#convert x_train and y_train to numpy arrays
x_train_NFLX,y_train_NFLX=np.array(x_train_NFLX),np.array(y_train_NFLX)
#reshape the data to 3 dimension
x_train_NFLX=np.reshape(x_train_NFLX,(x_train_NFLX.shape[0],x_train_NFLX.shape[1],1))


#split the AAPL_data into x_train and y_train dataset
#Incorporating Timesteps Into Data
x_train_AAPL, y_train_AAPL= [], []
for i in range(60,len(train_AAPL)):
    x_train_AAPL.append(scaled_data_AAPL[i - 60:i, 0])
    y_train_AAPL.append(scaled_data_AAPL[i, 0])
#convert x_train and y_train to numpy arrays
x_train_AAPL, y_train_AAPL= np.array(x_train_AAPL), np.array(y_train_AAPL)
#reshape the data to 3 dimension
x_train_AAPL=np.reshape(x_train_AAPL, (x_train_AAPL.shape[0], x_train_AAPL.shape[1], 1))


#split the META_data into x_train and y_train dataset
#Incorporating Timesteps Into Data
x_train_META, y_train_META= [], []
for i in range(60,len(train_META)):
    x_train_META.append(scaled_data_META[i - 60:i, 0])
    y_train_META.append(scaled_data_META[i, 0])
#convert x_train and y_train to numpy arrays
x_train_META, y_train_META= np.array(x_train_META), np.array(y_train_META)
#reshape the data to 3 dimension
x_train_META=np.reshape(x_train_META, (x_train_META.shape[0], x_train_META.shape[1], 1))


#split the MSFT_data into x_train and y_train dataset
#Incorporating Timesteps Into Data
x_train_MSFT, y_train_MSFT= [], []
for i in range(60,len(train_MSFT)):
    x_train_MSFT.append(scaled_data_MSFT[i - 60:i, 0])
    y_train_MSFT.append(scaled_data_MSFT[i, 0])
#convert x_train and y_train to numpy arrays
x_train_MSFT, y_train_MSFT= np.array(x_train_MSFT), np.array(y_train_MSFT)
#reshape the data to 3 dimension
x_train_MSFT=np.reshape(x_train_MSFT, (x_train_MSFT.shape[0], x_train_MSFT.shape[1], 1))


#split the TSLA_data into x_train and y_train dataset
#Incorporating Timesteps Into Data
x_train_TSLA, y_train_TSLA= [], []
for i in range(60,len(train_TSLA)):
    x_train_TSLA.append(scaled_data_TSLA[i - 60:i, 0])
    y_train_TSLA.append(scaled_data_TSLA[i, 0])
#convert x_train and y_train to numpy arrays
x_train_TSLA, y_train_TSLA= np.array(x_train_TSLA), np.array(y_train_TSLA)
#reshape the data to 3 dimension
x_train_TSLA=np.reshape(x_train_TSLA, (x_train_TSLA.shape[0], x_train_TSLA.shape[1], 1))

#load different model
model_NFLX=load_model("saved_model_NFLX.h5")
model_AAPL=load_model("saved_model_AAPL.h5")
model_META=load_model("saved_model_META.h5")
model_MSFT=load_model("saved_model_MSFT.h5")
model_TSLA=load_model("saved_model_TSLA.h5")


#create NFLX dataset x_test
inputs_NFLX=new_data_NFLX[len(new_data_NFLX)-len(valid_NFLX)-60:].values
inputs_NFLX=inputs_NFLX.reshape(-1,1)
inputs_NFLX=scaler.transform(inputs_NFLX)
X_test_NFLX=[]
for i in range(60,inputs_NFLX.shape[0]):
    X_test_NFLX.append(inputs_NFLX[i-60:i,0])
#convert data to numpy array
X_test_NFLX=np.array(X_test_NFLX)
#reshape the data
X_test_NFLX=np.reshape(X_test_NFLX,(X_test_NFLX.shape[0],X_test_NFLX.shape[1],1))

#get the models predicted price values
closing_price_NFLX=model_NFLX.predict(X_test_NFLX)
closing_price_NFLX=scaler.inverse_transform(closing_price_NFLX)
train_NFLX=new_data_NFLX[:1007]
valid_NFLX=new_data_NFLX[1007:]
valid_NFLX['Predictions']=closing_price_NFLX



#create AAPL dataset x_test
inputs_AAPL= new_data_AAPL[len(new_data_META) - len(valid_AAPL) - 60:].values
inputs_AAPL=inputs_AAPL.reshape(-1, 1)
inputs_AAPL=scaler.transform(inputs_AAPL)
X_test_AAPL=[]
for i in range(60, inputs_AAPL.shape[0]):
    X_test_AAPL.append(inputs_AAPL[i - 60:i, 0])
#convert data to numpy array
X_test_AAPL=np.array(X_test_AAPL)
#reshape the data
X_test_AAPL=np.reshape(X_test_AAPL, (X_test_AAPL.shape[0], X_test_AAPL.shape[1], 1))

#get the models predicted price values
closing_price_AAPL=model_AAPL.predict(X_test_AAPL)
closing_price_AAPL=scaler.inverse_transform(closing_price_AAPL)
train_AAPL= new_data_AAPL[:1007]
valid_AAPL= new_data_AAPL[1007:]
valid_AAPL['Predictions']=closing_price_AAPL



#create META dataset x_test
inputs_META= new_data_META[len(new_data_META) - len(valid_META) - 60:].values
inputs_META=inputs_META.reshape(-1, 1)
inputs_META=scaler.transform(inputs_META)
X_test_META=[]
for i in range(60, inputs_META.shape[0]):
    X_test_META.append(inputs_META[i - 60:i, 0])
#convert data to numpy array
X_test_META=np.array(X_test_META)
#reshape the data
X_test_META=np.reshape(X_test_META, (X_test_META.shape[0], X_test_META.shape[1], 1))

#get the models predicted price values
closing_price_META=model_META.predict(X_test_META)
closing_price_META=scaler.inverse_transform(closing_price_META)
train_META= new_data_META[:1007]
valid_META= new_data_META[1007:]
valid_META['Predictions']=closing_price_META



#create MSFT dataset x_test
inputs_MSFT= new_data_MSFT[len(new_data_MSFT) - len(valid_MSFT) - 60:].values
inputs_MSFT=inputs_MSFT.reshape(-1, 1)
inputs_MSFT=scaler.transform(inputs_MSFT)
X_test_MSFT=[]
for i in range(60, inputs_MSFT.shape[0]):
    X_test_MSFT.append(inputs_MSFT[i - 60:i, 0])
#convert data to numpy array
X_test_MSFT=np.array(X_test_MSFT)
#reshape the data
X_test_MSFT=np.reshape(X_test_MSFT, (X_test_MSFT.shape[0], X_test_MSFT.shape[1], 1))

#get the models predicted price values
closing_price_MSFT=model_MSFT.predict(X_test_MSFT)
closing_price_MSFT=scaler.inverse_transform(closing_price_MSFT)
train_MSFT= new_data_MSFT[:1007]
valid_MSFT= new_data_MSFT[1007:]
valid_MSFT['Predictions']=closing_price_MSFT




#create TSLA dataset x_test
inputs_TSLA= new_data_TSLA[len(new_data_TSLA) - len(valid_TSLA) - 60:].values
inputs_TSLA=inputs_TSLA.reshape(-1, 1)
inputs_TSLA=scaler.transform(inputs_TSLA)
X_test_TSLA=[]
for i in range(60, inputs_TSLA.shape[0]):
    X_test_TSLA.append(inputs_TSLA[i - 60:i, 0])
#convert data to numpy array
X_test_TSLA=np.array(X_test_TSLA)
#reshape the data
X_test_TSLA=np.reshape(X_test_TSLA, (X_test_TSLA.shape[0], X_test_TSLA.shape[1], 1))

#get the models predicted price values
closing_price_TSLA=model_TSLA.predict(X_test_TSLA)
closing_price_TSLA=scaler.inverse_transform(closing_price_TSLA)
train_TSLA= new_data_TSLA[:1007]
valid_TSLA= new_data_TSLA[1007:]
valid_TSLA['Predictions']=closing_price_TSLA

#combine all result to one DATASET
validData_NFLX = pd.DataFrame(valid_NFLX)
validData_NFLX['Stock'] = 'NFLX'
validData_NFLX['Date'] = valid_NFLX.index
validData_NFLX = validData_NFLX.reset_index(drop=True)

validData_AAPL = pd.DataFrame(valid_AAPL)
validData_AAPL['Stock'] = 'AAPL'
validData_AAPL['Date'] = valid_AAPL.index
validData_AAPL= validData_AAPL.reset_index(drop=True)

frames1 = [validData_AAPL, validData_NFLX]
result1 = pd.concat(frames1)

validData_META = pd.DataFrame(valid_META)
validData_META['Stock'] = 'META'
validData_META['Date'] = valid_META.index
validData_META= validData_META.reset_index(drop=True)

frames2 = [result1, validData_META]
result2 = pd.concat(frames2)

validData_MSFT = pd.DataFrame(valid_MSFT)
validData_MSFT['Stock'] = 'MSFT'
validData_MSFT['Date'] = valid_MSFT.index
validData_MSFT= validData_MSFT.reset_index(drop=True)

frames2 = [result2, validData_MSFT]
result3 = pd.concat(frames2)

validData_TSLA = pd.DataFrame(valid_TSLA)
validData_TSLA['Stock'] = 'TSLA'
validData_TSLA['Date'] = valid_TSLA.index
validData_TSLA= validData_TSLA.reset_index(drop=True)

frames3 = [result3, validData_TSLA]
validData = pd.concat(frames3)

#print(result4)


df= pd.read_csv("stock_data.csv")


# Data Visualization With Graphs
# set app layout
app.layout = html.Div([
    html.Br(),
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),
    html.Br(),
    dcc.Tabs(id="tabs", children=[
        dcc.Tab(label='Company LSTM model',children=[
            html.Div([
                html.Br(),
                html.H2("Actual closing price",style={"textAlign": "center"}),
                dcc.Dropdown(id='company-dropdown',
                             options=[{'label': 'Netflix', 'value': 'NFLX'},
                                      {'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'META', 'value': 'META'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['NFLX'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(
                    id="Actual Data",
                    # figure={
                    #     "data":[
                    #         go.Scatter(
                    #             x=valid.index,
                    #             y=valid["Close"],
                    #             mode='markers'
                    #         )
                    #     ],
                    #     "layout":go.Layout(
                    #         title='scatter plot',
                    #         xaxis={'title':'Date'},
                    #         yaxis={'title':'Closing Rate'}
                    #     )
                    #
                    # }
                ),
                html.H2("LSTM Predicted closing price",style={"textAlign": "center"}),
                dcc.Dropdown(id='company-dropdown2',
                             options=[{'label': 'Netflix', 'value': 'NFLX'},
                                      {'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple', 'value': 'AAPL'},
                                      {'label': 'META', 'value': 'META'},
                                      {'label': 'Microsoft', 'value': 'MSFT'}],
                             multi=True, value=['NFLX'],
                             style={"display": "block", "margin-left": "auto",
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(
                    id="Predicted Data",
                   # figure={
                        # "data":[
                        #     go.Scatter(
                        #         x=valid_NFLX.index,
                        #         y=valid_NFLX["Predictions"],
                        #         mode='markers'
                        #     )
                        # ],
                        # "layout":go.Layout(
                        #     title='scatter plot',
                        #     xaxis={'title':'Date'},
                        #     yaxis={'title':'Closing Rate'}
                        # )
                   # }
                )                
            ])                
        ]),
        dcc.Tab(label='Company Stock Data', children=[
            html.Div([
                html.Br(),
                html.H1("Company Stocks High vs Lows",
                        style={'textAlign': 'center'}),

                dcc.Dropdown(id='my-dropdown',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'META', 'value': 'META'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['META'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='highlow'),
                html.H1("Company Market Volume", style={'textAlign': 'center'}),
         
                dcc.Dropdown(id='my-dropdown2',
                             options=[{'label': 'Tesla', 'value': 'TSLA'},
                                      {'label': 'Apple','value': 'AAPL'}, 
                                      {'label': 'META', 'value': 'META'},
                                      {'label': 'Microsoft','value': 'MSFT'}], 
                             multi=True,value=['META'],
                             style={"display": "block", "margin-left": "auto", 
                                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='volume')
            ], className="container"),
        ])
    ])
])
# callbacks
@app.callback(Output('Actual Data', 'figure'),
              [Input('company-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","META": "META","MSFT": "Microsoft", "NFLX":"Netflix"}
    trace1 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=validData[validData["Stock"] == stock]["Date"],
                     y=validData[validData["Stock"] == stock]["Close"],
                     mode='lines+markers',
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual Close Price for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 1, 'label': 'YTD',
                                                       'step': 'year',
                                                       'stepmode': 'todate'},
                                                      {'count': 1, 'label': '1Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 3, 'label': '3Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 5, 'label': '5Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Close Rate (USD)"})}
    return figure


@app.callback(Output('Predicted Data', 'figure'),
              [Input('company-dropdown2', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","META": "META","MSFT": "Microsoft", "NFLX":"Netflix"}
    trace1 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=validData[validData["Stock"] == stock]["Date"],
                     y=validData[validData["Stock"] == stock]["Predictions"],
                     mode='lines+markers',
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1',
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Actual Close Price for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 1, 'label': 'YTD',
                                                       'step': 'year',
                                                       'stepmode': 'todate'},
                                                      {'count': 1, 'label': '1Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 3, 'label': '3Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 5, 'label': '5Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Close Rate (USD)"})}
    return figure
@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","META": "META","MSFT": "Microsoft",}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["High"],
                     mode='lines', opacity=0.7, 
                     name=f'High {dropdown[stock]}',textposition='bottom center'))
        trace2.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Low"],
                     mode='lines', opacity=0.6,
                     name=f'Low {dropdown[stock]}',textposition='bottom center'))
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"High and Low Prices for {', '.join(str(dropdown[i]) for i in selected_dropdown)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month', 
                                                       'stepmode': 'backward'},
                                                      {'count': 1, 'label': 'YTD',
                                                       'step': 'year',
                                                       'stepmode': 'todate'},
                                                      {'count': 1, 'label': '1Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 3, 'label': '3Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 5, 'label': '5Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Price (USD)"})}
    return figure
# callbacks
@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2', 'value')])
def update_graph(selected_dropdown_value):
    dropdown = {"TSLA": "Tesla","AAPL": "Apple","META": "META","MSFT": "Microsoft",}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
          go.Scatter(x=df[df["Stock"] == stock]["Date"],
                     y=df[df["Stock"] == stock]["Volume"],
                     mode='lines', opacity=0.7,
                     name=f'Volume {dropdown[stock]}', textposition='bottom center'))

    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data, 
              'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                            '#FF7400', '#FFF400', '#FF0056'],
            height=600,
            title=f"Market Volume for {', '.join(str(dropdown[i]) for i in selected_dropdown_value)} Over Time",
            xaxis={"title":"Date",
                   'rangeselector': {'buttons': list([{'count': 1, 'label': '1M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 6, 'label': '6M',
                                                       'step': 'month',
                                                       'stepmode': 'backward'},
                                                      {'count': 1, 'label': 'YTD',
                                                       'step': 'year',
                                                       'stepmode': 'todate'},
                                                      {'count': 1, 'label': '1Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 3, 'label': '3Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'count': 5, 'label': '5Y',
                                                       'step': 'year',
                                                       'stepmode': 'backward'},
                                                      {'step': 'all'}])},
                   'rangeslider': {'visible': True}, 'type': 'date'},
             yaxis={"title":"Transactions Volume"})}
    return figure
# telling our app to start the server if we are running this file
if __name__=='__main__':
    app.run_server(debug=True)