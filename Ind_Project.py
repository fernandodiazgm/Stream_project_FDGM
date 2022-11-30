# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 16:17:05 2022

@author: fdiazgonzalezmanja
"""
#Notas para futuras modificaciones
#qUITAR EL INDEX DE LAS TABLAS
#PONER MAS INFORMACION EN LAS TABLAS

#Import all the libraries that are going to be used in the code
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
import matplotlib.pyplot as plt
import pandas_datareader.data as web
import datetime as dt
import mplfinance as mpf

#Set main title of the project
st.title("Individual project Python Analysis on S&P 500")
st.write("Fernando DIAZ GONZALEZ MANJARREZ")

#Create tabs
tab1, tab2, tab3, tab4, tab5=st.tabs(["Summary","Chart","Financials","Monte Carlo Simulation", "Top tickers"])

#Get list of S&P 500 tickers
ticker_list = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]['Symbol']

#--------------------------------------------------------------------------------------
# FIRST TAB
#---------------------------------------------------------------------------------------

with tab1:

    #Create selection box for the user to select only one
    ticker = st.selectbox("Ticker(s)", ticker_list)
    
    #Create date input for the user to select dates
    col1, col2=st.columns(2)
    date1 = col1.date_input("Start date", datetime.today().date() - timedelta(days=30))
    
    #Create selectbox for the user to select the date frame
    graph_aux=("1M","3M","6M","YTD","1Y","5Y","Max")
    date_aux = st.selectbox("View of the graph", graph_aux)
    
    #Add button to get the data
    get = st.button("Get data", key="get")
    
    #Define function to get the first table
    def get_t1(t1):
        if len(t1)>0:
            
            #Create an empty dataframe with the information it will have
            concepts=("Previous Close", "Open", "Day's Range", "52 week Range","Volume")
            ini_val=(0,0,0,0,0)
            tab1=pd.DataFrame()
            tab1["Concept"]=concepts
            tab1["Values"]=ini_val
            
            
            #Get the first info// by default we will extract 10 days only to get the information required for this table
            date2=date1-timedelta(days=10)
            info1 = yf.Ticker(t1).history(start=date2, end=date1).loc[:,['Open','High','Low','Close','Volume']]
            
            #Get previous close
            previous_close=info1.loc[:,"Close"][-2]
            tab1.loc[0,"Values"]=round(previous_close,2)
            
            #Get Open information
            open=info1.loc[:,"Open"][-1]
            tab1.loc[1,"Values"]=round(open,2)
            
            #Get rank of values
            high=round(info1.loc[:,"High"][-1],2)
            low=round(info1.loc[:,"Low"][-1],2)
            dr=str(high)+"-"+str(low)
            tab1.loc[2,"Values"]=dr
            
            #Get volume
            volume=info1.loc[:,"Volume"][-1]
            tab1.loc[4,"Values"]=volume
            
            #Get the 52 weeks info
            d1=date1
            d2=d1-timedelta(weeks=52)
            values1 = yf.Ticker(t1).history(start=d2, end=d1)
            #Get the max value and min in 52 weeks
            high=round(values1.loc[:,"High"],2)
            low=round(values1.loc[:,"Low"],2)
            m1=max(high)
            m2=min(low)
            #Concatenate the the values
            range52=str(m2)+" - "+str(m1)
            #Add it to the table
            tab1.loc[3,"Values"]=range52
            
            #Change the structure of the table so that it is displayed nicely
            tab1.index=tab1.iloc[:,0]
            tab1=tab1.iloc[:,1]
            
            #Print table1
            st.title("Information about the value of the stock")
            st.dataframe(tab1)
            
    #Function to get the profile of the company and the major holders
    def get_info(t2):
        if len(t2)>0:
            info2=yf.Ticker(t2)
            
            #Business Summary
            bs=info2.info["longBusinessSummary"]
            
            #Major Holders
            mh=info2.major_holders
            mh.columns=['Share','Concept']
            
            
            #Create table for the profile of the company
            col_names1=("Long Name", "Country", "Exchange time zone","Industry", "Sector")
            tab2=pd.DataFrame()
            tab2["Concept"]=col_names1
            tab2.loc[0," "]=info2.info["longName"]
            tab2.loc[1," "]=info2.info["country"]
            tab2.loc[2," "]=info2.info["exchangeTimezoneName"]
            tab2.loc[3," "]=info2.info["industry"]
            tab2.loc[4," "]=info2.info["sector"]
            
            #Change the structure of both tables so that they are displayed nicely
            tab2.index=tab2.iloc[:,0]
            tab2=tab2.iloc[:,1]
            mh.index=mh.iloc[:,1]
            mh=mh.iloc[:,0]
            
            #Print the info
            st.title("Profile of the company")
            st.dataframe(tab2)
            st.title("Major Shareholders")
            st.dataframe(mh)
            st.title("Summary of the company")
            st.write(bs)
    #Function to plot the information       
    def get_plot(t3,date1):
        
        if len(t3)>0:
            aux_max=0    
            
            #Get the the end date depending on the button
            
            if date_aux=="1M":
                date3=date1-timedelta(days=30)
            elif date_aux=="3M":
                date3=date1-timedelta(days=30*3)
            elif date_aux=="6M":
                date3=date1-timedelta(days=30*6)
            elif date_aux=='YTD':
                date3=datetime(date1.year,1,1)
            elif date_aux=='1Y':
                date3=date1-timedelta(weeks=52*1)
            elif date_aux=='5Y':
                date3=date1-timedelta(weeks=52*5)
            elif date_aux=='Max':
                aux_max=1
            else:
                date3=date1-timedelta(days=30)
            
            #Get info
            if aux_max==1:
               info3 = yf.Ticker(t3).history(period='max')
            else:
                info3 = yf.Ticker(t3).history(start=date3, end=date1)
            
            #Create plot
            fig, ax=mpf.plot(info3, type='line', style='classic', returnfig=True, volume=True, datetime_format='%m-%d-%Y')                
            
            #Show the plot
            st.pyplot(fig)
        
            
    if get:
        get_t1(ticker[0])
        get_plot(ticker[0],date1)
        get_info(ticker[0])

#--------------------------------------------------------------------------------------
# SECOND TAB
#---------------------------------------------------------------------------------------

with tab2:
    #Create selection box for the user to select only one
    ticker2 = st.selectbox("Ticker(s)  ", ticker_list)
    
    #Create selection box for the user to select the range of dates
    col12, col22=st.columns(2)
    date12 = col12.date_input("Start date ", datetime.today().date() - timedelta(days=30))
    date22 = col22.date_input("End date ", datetime.today().date())
    
    #Select the range of days to see
    graph_aux=("No filter","1M","3M","6M","YTD","1Y","5Y","Max")
    date_aux = st.selectbox("View of the graph", graph_aux)
    
    #Select view of the graph
    time_intervals=("Days","Month","Year")
    time_aux = st.selectbox("View of the graph ", time_intervals)
    
    #Select type of plot 
    graph_t=("Line plot","Candle plot")
    graph_taux = st.selectbox("Type of graph ", graph_t)
    
    #Add button to get the data
    get2 = st.button("Get data ", key="get2")
    
    #Function to plot the information 
    def get_plot2(t3,date1,date2):
        
        if len(t3)>0:
            aux_max=0    
            
            #Get the start date depending on the button
            if date_aux=="1M":
                date1=date2-timedelta(days=30)
            elif date_aux=="3M":
                date1=date2-timedelta(days=30*3)
            elif date_aux=="6M":
                date1=date2-timedelta(days=30*6)
            elif date_aux=='YTD':
                date1=datetime(date2.year,1,1)
            elif date_aux=='1Y':
                date1=date2-timedelta(weeks=52*1)
            elif date_aux=='5Y':
                date1=date2-timedelta(weeks=52*5)
            elif date_aux=='Max':
                aux_max=1
            
            #Get the maximum record to calcualte the moving average of 50 days even if another range is selected
            info3 = yf.Ticker(t3).history(period='max').loc[:,['Close','Volume','Open','High','Low']]
            
            #Calculate the moving average
            info3['Moving average']=info3.loc[:,'Close'].rolling(50).mean()
            
            #Get year month and get the index in a column that will be used later
            info3['year']=info3.index.year
            info3['month']=info3.index.month
            info3['date1']=info3.index
            
            #Create new column and convert it to a datetime format so the df can be filtered later
            info3['date2']=pd.to_datetime(info3.index.strftime("%Y-%m-%d"),format='%Y-%m-%d')
            
            #Subset the data only if there is no max selected
            if aux_max!=1:
                
                #Subset the data if the maximum is not wanted 
                info4=yf.Ticker(t3).history(start=date1, end=date2).loc[:,['Close','Volume','Open','High','Low']]
                
                #Merge with the max data frame created above to get the moving average
                #Get the column date1 that will help us to merge the tables
                info4['date1']=info4.index
                #Merge the data frames
                info3=pd.merge(info4, info3,how='left',on='date1')
                #Rename the index
                info3.index=info3.loc[:,'date1']
                #Remove duplicated columns
                info3=info3.iloc[:,-10:]
                #Rename columns
                info3.columns=['date1', 'Close', 'Volume', 'Open', 'High', 'Low','Moving average', 'year', 'month', 'date2']
                
            #Subset the basetable to get the view for months or year
            date_f='%m-%d-%Y'
            if time_aux=='Month': 
                info3=info3.groupby(['year','month'], as_index=False).last()
                info3.index=pd.to_datetime(info3['date1'])
                date_f='%b-%Y'
                
            elif time_aux=='Year':
                info3=info3.groupby(['year'], as_index=False).last()
                info3.index=pd.to_datetime(info3['date1'])
                date_f='%Y'
            
            
            #Create plot depending if it is line or it is candle plot
            fig, ax = plt.subplots(figsize=(20, 3))
            
            if graph_taux=='Line plot':
                #Create plots with matplotfinance
                ma=mpf.make_addplot(info3.loc[:,'Moving average'],type='line')                
                fig, ax=mpf.plot(info3, type='line', style='classic', addplot=ma, returnfig=True, volume=True, datetime_format=date_f)                
                st.pyplot(fig)
                
            elif graph_taux=='Candle plot':
                
                #Create plots with matplotfinance
                ma=mpf.make_addplot(info3.loc[:,'Moving average'],type='line')                
                fig, ax=mpf.plot(info3, type='candle', style='classic', addplot=ma, returnfig=True, volume=True, datetime_format=date_f)               
                st.pyplot(fig)
                
                

    #Put condition to call functions
    if get2:
        get_plot2(ticker2[0],date12, date22)
        st.text('*The blue line represents the moving average of 50 days ')
        
        
#--------------------------------------------------------------------------------------
# THIRD TAB
#---------------------------------------------------------------------------------------

with tab3:
    #Get the ticker from the user
    ticker3 = st.selectbox("Ticker(s)    ", ticker_list)
    
    #Let the user decide the type of time view to see
    time_fin=("Quarterly","Annualy")
    time_fin1 = st.selectbox("Time view ", time_fin)
    
    #Buttons to select the type of report the user wants to see
    ginc = st.button("Income statement ", key="gincome")
    gbal = st.button("Balance statement ", key="gbalance")
    gcas = st.button("Cash flow", key="gcash")
    
    
    
    #Get the information of the ticker
    info5=yf.Ticker(ticker3[0])
    
    #Set the table for the Income Statement
    if ginc:
        #Get the financial data
        if time_fin1=='Annualy':
            info5=info5.financials
        elif time_fin1=='Quarterly':
            info5=info5.quarterly_financials
        #Get columns that going to be displayed
        info5=info5.loc[['Total Revenue','Cost Of Revenue','Gross Profit','Total Operating Expenses',
         'Selling General Administrative','Research Development','Operating Income',
         'Total Other Income Expense Net', 'Income Before Tax','Income Tax Expense',
        'Net Income','Operating Income','Total Operating Expenses','Net Income From Continuing Ops',
         'Interest Expense','Ebit'],:]
        
        #Give the desired format to the values
        #Convert name of the columns to string, since it has a datetime format
        info5.columns=info5.columns.astype(str)
        
        #Cycle thorugh each column and divide the values by 1000
        for i in info5.columns:
            info5[i]=info5.loc[:,i]/1000
        
        #Give the desired format to the table
        info5=info5.style.format("{:,.0f}")
        
        #Display the table
        st.title(time_fin1+" Income Statement ")
        st.dataframe(info5)
    
    #Set the table for the Balance Sheet
    if gbal:
        
        #Get the Balance Statement data -----> SELECT THE CORRECT COLUMNS
        if time_fin1=='Annualy':
            info5=info5.balance_sheet
        elif time_fin1=='Quarterly':
            info5=info5.quarterly_balance_sheet
        
        #Get columns that going to be displayed
        info5=info5.loc[['Total Assets','Total Current Assets','Total Liab','Total Stockholder Equity','Net Tangible Assets'],:]
        
        #Give the desired format to the values
        #Convert name of the columns to string, since it has a datetime format
        info5.columns=info5.columns.astype(str)
        #Cycle thorugh each column and divide the values by 1000
        for i in info5.columns:
            info5[i]=info5.loc[:,i]/1000
        #Give the diserd format to the table
        info5=info5.style.format("{:,.0f}")
        
        #Display the table
        st.title(time_fin1+" Balance Sheet ")
        st.dataframe(info5)
    
    if gcas:
        #Get the Balance Statement data
        if time_fin1=='Annualy':
            info5=info5.cashflow
        elif time_fin1=='Quarterly':
            info5=info5.quarterly_cashflow
        
        #Get columns that going to be displayed
        info5=info5.loc[['Total Cash From Operating Activities','Total Cashflows From Investing Activities',
         'Total Cash From Financing Activities','Capital Expenditures','Issuance Of Stock'],:]
        
        #Give the diserd format to the values
        #Convert name of the columns to string, since it has a datetime format
        info5.columns=info5.columns.astype(str)
        #Cycle thorugh each column and divide the values by 1000
        for i in info5.columns:
            info5[i]=info5.loc[:,i]/1000
        #Give the diserd format to the table float with no decimals and comma separator
        info5=info5.style.format("{:,.0f}")
        
        #Display the table
        st.title(time_fin1+" Cash Flow ")
        st.dataframe(info5)

#--------------------------------------------------------------------------------------
# DEFINITION OF CLASS MONTECARLO
#---------------------------------------------------------------------------------------


class MonteCarlo(object):
    
    def __init__(self, ticker, data_source, start_date, end_date, time_horizon, n_simulation, seed):
        
        # Initiate class variables
        self.ticker = ticker  # Stock ticker
        self.data_source = data_source  # Source of data, e.g. 'yahoo'
        self.start_date = dt.datetime.strptime(start_date, '%Y-%m-%d')  # Text, YYYY-MM-DD
        self.end_date = dt.datetime.strptime(end_date, '%Y-%m-%d')  # Text, YYYY-MM-DD
        self.time_horizon = time_horizon  # Days
        self.n_simulation = n_simulation  # Number of simulations
        self.seed = seed  # Random seed
        self.simulation_df = pd.DataFrame()  # Table of results
        
        # Extract stock data
        self.stock_price = web.DataReader(ticker, data_source, self.start_date, self.end_date)
        
        # Calculate financial metrics
        # Daily return (of close price)
        self.daily_return = self.stock_price['Close'].pct_change()
        # Volatility (of close price)
        self.daily_volatility = np.std(self.daily_return)
        
    def run_simulation(self):
        
        # Run the simulation
        np.random.seed(self.seed)
        self.simulation_df = pd.DataFrame()  # Reset
        
        for i in range(self.n_simulation):

            # The list to store the next stock price
            next_price = []

            # Create the next stock price
            last_price = self.stock_price['Close'][-1]

            for j in range(self.time_horizon):
                
                # Generate the random percentage change around the mean (0) and std (daily_volatility)
                future_return = np.random.normal(0, self.daily_volatility)

                # Generate the random future price
                future_price = last_price * (1 + future_return)

                # Save the price and go next
                next_price.append(future_price)
                last_price = future_price

            # Store the result of the simulation
            next_price_df = pd.Series(next_price).rename('sim' + str(i))
            self.simulation_df = pd.concat([self.simulation_df, next_price_df], axis=1)

    def plot_simulation_price(self):
        
        # Plot the simulation stock price in the future
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 10, forward=True)

        plt.plot(self.simulation_df)
        plt.title('Monte Carlo simulation for ' + self.ticker + \
                  ' stock price in next ' + str(self.time_horizon) + ' days')
        plt.xlabel('Day')
        plt.ylabel('Price')

        plt.axhline(y=self.stock_price['Close'][-1], color='red')
        plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')

        return plt
    
    def plot_simulation_hist(self):
        
        # Get the ending price of the 200th day
        ending_price = self.simulation_df.iloc[-1:, :].values[0, ]

        # Plot using histogram
        fig, ax = plt.subplots()
        plt.hist(ending_price, bins=50)
        plt.axvline(x=self.stock_price['Close'][-1], color='red')
        plt.legend(['Current stock price is: ' + str(np.round(self.stock_price['Close'][-1], 2))])
        ax.get_legend().legendHandles[0].set_color('red')
        plt.show()
    
    def value_at_risk(self):
        # Price at 95% confidence interval
        future_price_95ci = np.percentile(self.simulation_df.iloc[-1:, :].values[0, ], 5)

        # Value at Risk
        VaR = self.stock_price['Close'][-1] - future_price_95ci
        VaR=str(np.round(VaR, 2))
        return  VaR
        

       
#--------------------------------------------------------------------------------------
# FOURTH TAB
#---------------------------------------------------------------------------------------

with tab4:
    #Get the ticker from the user
    ticker4 = st.selectbox("Ticker(s)     ", ticker_list)
    
    #Let the user decide the type of time view to see
    sim_aux=(200,500,1000)
    num_sim = st.selectbox("Select the amount of simulations ", sim_aux)
    hor_aux=(30,60,90)
    day_hor = st.selectbox("Select the horizon (days) of the simulation ", hor_aux)
    
    #Create button to make the simulation
    sim_b = st.button("Start simulation ", key="sim_k")
    
    def get_simulation(num_sim,day_hor):
        #The end date will always be today
        today=date.today()
        
        #The start date for the montecarlo simulation will be default 60 days
        start_date=today-timedelta(days=60)
        
        #Initialize MonteCarlo
        
        mc_sim = MonteCarlo(ticker=ticker4[0], data_source='yahoo',
                        start_date=str(start_date), end_date=str(today),
                        time_horizon=day_hor, n_simulation=num_sim, seed=123)
        # Run simulation
        mc_sim.run_simulation()
        
        # Plot the results
        st.pyplot(mc_sim.plot_simulation_price())
        
        #Get the risk of the value at 95% confidence
        y=mc_sim.value_at_risk()
    
        #Display the Value at Risk
        st.title('Value at risk at 95% confidence interval is: '+y+' USD')
    
    if sim_b:
        get_simulation(num_sim,day_hor)

        
#--------------------------------------------------------------------------------------
# FIFTH TAB
#---------------------------------------------------------------------------------------

with tab5:
    
    
    col15, col25=st.columns(2)
    date15 = col15.date_input("Start date     ", datetime.today().date() - timedelta(days=30))
    date25 = col25.date_input("End date      ", datetime.today().date())

    aux_max=0    
    #Select view of the graph
    graph_aux=("Range from above","1M","3M","6M","YTD","1Y","5Y","Max")
    date_aux = st.selectbox("Select time frame", graph_aux)
    if date_aux=="1M":
        date15=date25-timedelta(days=30)
    elif date_aux=="3M":
        date15=date25-timedelta(days=30*3)
    elif date_aux=="6M":
        date15=date25-timedelta(days=30*6)
    elif date_aux=='YTD':
        date15=datetime(date25.year,1,1)
    elif date_aux=='1Y':
        date15=date25-timedelta(weeks=52*1)
    elif date_aux=='5Y':
        date15=date25-timedelta(weeks=52*5)
    elif date_aux=='Max':
        aux_max=1
    
    
    plus_minus=("more return", "more loss")
    
    plus_aux = st.selectbox("Select if you want to see the tickers with more or less return per stock bought", plus_minus)
    
    top=('All','10','20','30','40','50')
    
    top_aux = st.selectbox("Select the number of rows you want to show", top)
    

    # --- Add a buttons ---
    
    get5 = st.button("Get data      ", key="get_5")
    
    def t5(date15,date25,top_aux):
        
        num_t=0
        info6 = pd.DataFrame()
      
        progress=st.progress(0)
        
        
        
        for tick in ticker_list:
            
            
            try:    
                if aux_max==1:
                    stock_df = yf.Ticker(tick).history(period=max)
                else:
                    stock_df = yf.Ticker(tick).history(start=date15, end=date25)
                stock_df=stock_df.iloc[[0,-1]]
                stock_df['last_price']=stock_df.loc[:,'Close'].shift(1) #Get the last result
                stock_df=stock_df.loc[:,['Close','last_price','Volume']]
                stock_df=stock_df.iloc[1:2] #Select last row
                stock_df['Change_price']=stock_df['last_price']-stock_df['Close']
                stock_df['Change_price_p']=stock_df['last_price']/stock_df['Close']-1
                stock_df['name_1']=tick
                info6 = pd.concat([info6, stock_df], axis=0) # Combine results
                num_t+=1.0/len(ticker_list)
                
            except Exception:
                pass
            
            progress.progress(num_t)
           
        info6=info6.loc[:,['name_1','Close','Change_price','Change_price_p','Volume']]
        if plus_aux=='more return':
            info6=info6.sort_values(['Change_price'],ascending=False)
        elif plus_aux=='more loss':
            info6=info6.sort_values(['Change_price'],ascending=True)
        if top_aux !='All':
            top_aux1=int(top_aux)
            info6=info6.iloc[0:top_aux1]
        top_ind=len(info6)+1
        info6.index=range(1,top_ind)
        info6.columns=['Ticker name', 'Close price','Change in price $','Change in price %','Volume']
        info6=info6.style.format({'Close price':'{:,.2f}','Change in price $':'{:,.2f}','Change in price %':'{:.2%}', 'Volume':'{:,.0f}'})
        
        
        
        
        #Print the table
        st.title("Stocks ordered with " + plus_aux)
        st.dataframe(info6)
        
        
        

    if get5:
        
        t5(date15,date25,top_aux)
        


        
        
        
    
        
        