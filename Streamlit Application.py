# -*- coding: utf-8 -*-
"""
Created on Sat Oct 30 14:41:37 2021

@author: hhussain1
"""



import yahoo_fin.stock_info as si
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import yfinance as yf
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#import pyfolio as pf



   
    
#==============================================================================
# Tab 1 Summary
#==============================================================================

def tab1():
    
    st.title("Summary")
    st.write("Select ticker on the left to begin")
    st.write(ticker)
    
    #The code below gets the quota table from Yahoo Finance. The streamlit page
    #is divided into 2 columns and selected columns are displayed on each side of the page.

    def getsummary(ticker):
            table = si.get_quote_table(ticker, dict_result = False)
            return table 
        
    c1, c2 = st.columns((1,1))
    with c1:        
        if ticker != '-':
            summary = getsummary(ticker)
            summary['value'] = summary['value'].astype(str)
            showsummary = summary.iloc[[14, 12, 5, 2, 6, 1, 16, 3],]
            showsummary.set_index('attribute', inplace=True)
            st.dataframe(showsummary)
            
            
    with c2:        
        if ticker != '-':
            summary = getsummary(ticker)
            summary['value'] = summary['value'].astype(str)
            showsummary = summary.iloc[[11, 4, 13, 7, 8, 10, 9, 0],]
            showsummary.set_index('attribute', inplace=True)
            st.dataframe(showsummary)
            
                        
             
    #The code below uses the yahoofinance package to get all the available stock
    #price data. Plotly is then used to visualize the data.  An interesting feature
    #from plotly called range selector is also used. A list of dictionaries
    #is added to range selector to make buttons and identify the periods.
    #References:
    #https://plotly.com/python/range-slider/
    
        
    @st.cache 
    def getstockdata(ticker):
        stockdata = yf.download(ticker, period = 'MAX')
        return stockdata
        
    if ticker != '-':
            chartdata = getstockdata(ticker) 
                       
            fig = px.area(chartdata, chartdata.index, chartdata['Close'])
            
                     

            fig.update_xaxes(
                rangeselector=dict(
                    buttons=list([
                        dict(count=1, label="1M", step="month", stepmode="backward"),
                        dict(count=3, label="3M", step="month", stepmode="backward"),
                        dict(count=6, label="6M", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1Y", step="year", stepmode="backward"),
                        dict(count=3, label="3Y", step="year", stepmode="backward"),
                        dict(count=5, label="5Y", step="year", stepmode="backward"),
                        dict(label = "MAX", step="all")
                    ])
                )
            )
            st.plotly_chart(fig)
            
     
              
    
#==============================================================================
# Tab 2 Chart
#==============================================================================


#The code below divides the streamlit page into 5 columns. The first two columns
#have a date picker option to select start and end dates and the the other three
#have dropdown selection boxes for duration, interval, and type of plot.

def tab2():
    st.title("Chart")
    st.write(ticker)
    
    st.write("Set duration to '-' to select date range")
    
    c1, c2, c3, c4,c5 = st.columns((1,1,1,1,1))
    
    with c1:
        
        start_date = st.date_input("Start date", datetime.today().date() - timedelta(days=30))
        
    with c2:
        
        end_date = st.date_input("End date", datetime.today().date())        
        
    with c3:
        
        duration = st.selectbox("Select duration", ['-', '1Mo', '3Mo', '6Mo', 'YTD','1Y', '3Y','5Y', 'MAX'])          
        
    with c4: 
        
        inter = st.selectbox("Select interval", ['1d', '1mo'])
        
    with c5:
        
        plot = st.selectbox("Select Plot", ['Line', 'Candle'])
        
 
#The code below first obtains all the data using the download option from yahoo finance.
#It then creates a column for the simple moving average, makes the date index into a column
#and then subsets the dataframe to get just the date and and SMA column.
#Then if a duration is selected from the dropdown, data for that duration is downloaded
# and the SMA column is merged to the dataframe. If a duration is not selected then
#automatically the specified date range is used to get the data and that is also merged
#with the SMA column
#References:
#https://towardsdatascience.com/data-science-in-finance-56a4d99279f7

           
             
    @st.cache             
    def getchartdata(ticker):
        SMA = yf.download(ticker, period = 'MAX')
        SMA['SMA'] = SMA['Close'].rolling(50).mean()
        SMA = SMA.reset_index()
        SMA = SMA[['Date', 'SMA']]
        
        if duration != '-':        
            chartdata1 = yf.download(ticker, period = duration, interval = inter)
            chartdata1 = chartdata1.reset_index()
            chartdata1 = chartdata1.merge(SMA, on='Date', how='left')
            return chartdata1
        else:
            chartdata2 = yf.download(ticker, start_date, end_date, interval = inter)
            chartdata2 = chartdata2.reset_index()
            chartdata2 = chartdata2.merge(SMA, on='Date', how='left')                             
            return chartdata2
    
#The code below uses plotly to visualize the data. Subplots from plotly is used to make 2 y axis.
#First y axis shows the stock close price and SMA and the second is used to show volume. 
#Plotly graph objects are used to add graphs to the axes.The range for the y axis for 
#volume is manipulated so that the bars appear small.
#References:
#https://plotly.com/python/multiple-axes/   
#https://plotly.com/python/candlestick-charts/        
        
    if ticker != '-':
            chartdata = getchartdata(ticker) 
            
                       
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            
            if plot == 'Line':
                fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['Close'], mode='lines', 
                                         name = 'Close'), secondary_y = False)
            else:
                fig.add_trace(go.Candlestick(x = chartdata['Date'], open = chartdata['Open'], 
                                             high = chartdata['High'], low = chartdata['Low'], close = chartdata['Close'], name = 'Candle'))
              
                    
            fig.add_trace(go.Scatter(x=chartdata['Date'], y=chartdata['SMA'], mode='lines', name = '50-day SMA'), secondary_y = False)
            
            fig.add_trace(go.Bar(x = chartdata['Date'], y = chartdata['Volume'], name = 'Volume'), secondary_y = True)

            fig.update_yaxes(range=[0, chartdata['Volume'].max()*3], showticklabels=False, secondary_y=True)
        
      
            st.plotly_chart(fig)
           
             

#==============================================================================
# Tab 3 Statistics
#==============================================================================

#The code below obtains information using get_stats_valuation and get_stats in
#Yahoo Finance. It then slices the dataframes and displays them in different 
#columns of the streamlit page under different headings.

def tab3():
     st.title("Statistics")
     st.write(ticker)
     c1, c2 = st.columns(2)
     
         
     
     with c1:
         st.header("Valuation Measures")
         #@st.cache
         def getvaluation(ticker):
                 return si.get_stats_valuation(ticker)
    
         if ticker != '-':
                valuation = getvaluation(ticker)
                valuation[1] = valuation[1].astype(str)
                valuation = valuation.rename(columns = {0: 'Attribute', 1: ''})
                valuation.set_index('Attribute', inplace=True)
                st.table(valuation)
                
        
         st.header("Financial Highlights")
         st.subheader("Fiscal Year")
         
         #@st.cache
         def getstats(ticker):
                 return si.get_stats(ticker)
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[29:31,])
                
        
         st.subheader("Profitability")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[31:33,])
                
                
                
         st.subheader("Management Effectiveness")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[33:35,])
         
         
                
         st.subheader("Income Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[35:43,])  
            
         
         st.subheader("Balance Sheet")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[43:49,])
         
         st.subheader("Cash Flow Statement")
         
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[49:,])
         
        
                           
     with c2:
         st.header("Trading Information")
         
         
         st.subheader("Stock Price History")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[:7,])
         
         st.subheader("Share Statistics")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[7:19,])
         
         st.subheader("Dividends & Splits")
                  
         if ticker != '-':
                stats = getstats(ticker)
                stats['Value'] = stats['Value'].astype(str)
                stats.set_index('Attribute', inplace=True)
                st.table(stats.iloc[19:29,])
         
         
         
            
     

#==============================================================================
# Tab 4 Financials
#==============================================================================

#The code below obtains yearly and quartely financial statements from Yahoo Finance
#and displays them according the options selected by the users in streamlit. A
#combination of if statements is used to display according to the selected options.


def tab4():
      st.title("Financials")
      st.write(ticker)
      
      statement = st.selectbox("Show", ['Income Statement', 'Balance Sheet', 'Cash Flow'])
      period = st.selectbox("Period", ['Yearly', 'Quarterly'])
      
      @st.cache
      def getyearlyincomestatement(ticker):
            return si.get_income_statement(ticker)
      
      @st.cache
      def getquarterlyincomestatement(ticker):
            return si.get_income_statement(ticker, yearly = False)
      
      @st.cache
      def getyearlybalancesheet(ticker):
            return si.get_balance_sheet(ticker)
      
      @st.cache
      def getquarterlybalancesheet(ticker):
            return si.get_balance_sheet(ticker, yearly = False)      

      @st.cache
      def getyearlycashflow(ticker):
            return si.get_cash_flow(ticker)
      
      @st.cache
      def getquarterlycashflow(ticker):
            return si.get_cash_flow(ticker, yearly = False)
        
          
      if ticker != '-' and statement == 'Income Statement' and period == 'Yearly':
                data = getyearlyincomestatement(ticker)
                st.table(data)
            
      if ticker != '-' and statement == 'Income Statement' and period == 'Quarterly':
                data = getquarterlyincomestatement(ticker)
                st.table(data)            

      if ticker != '-' and statement == 'Balance Sheet' and period == 'Yearly':
                data = getyearlybalancesheet(ticker)
                st.table(data)            
      
      if ticker != '-' and statement == 'Balance Sheet' and period == 'Quarterly':
                data = getquarterlybalancesheet(ticker)
                st.table(data)        
      
      if ticker != '-' and statement == 'Cash Flow' and period == 'Yearly':
                data = getyearlycashflow(ticker)
                st.table(data)        
      
        
      if ticker != '-' and statement == 'Cash Flow' and period == 'Quarterly':
                data = getquarterlycashflow(ticker)
                st.table(data)      
                
                 
        
      
        
      
#==============================================================================
# Tab 5 Analysis
#==============================================================================

#In the code below, get_analysts_info is used to obtain the data. The output is
#in the form of a dictionary. .items() is used to get the items from the dictionary
#and then a for loop i used under which the dictionary items are changed into a list
# and each element of the list is then converted to a dataframe for displaying.


def tab5():
      st.title("Analysis")
      st.write("Currency in USD")
      st.write(ticker)
      
      @st.cache
      def getanalysis(ticker):
            analysis_dict = si.get_analysts_info(ticker)
            return analysis_dict.items()
 
           
      if ticker != '-':           
           for i in range(6):
            analysis = getanalysis(ticker)
            df = pd.DataFrame(list(analysis)[i][1])
            st.table(df)
            
           
#==============================================================================
# Tab 6 Monte Carlo Simulation
#==============================================================================

#The code below performs and displays the monte carlo simulation for a specified
#time horizon and number of intervals



def tab6():
     st.title("Monte Carlo Simulation")
     st.write(ticker)
     
     #Dropdown for selecting simulation and horizon
     simulations = st.selectbox("Number of Simulations (n)", [200, 500, 1000])
     time_horizon = st.selectbox("Time Horizon (t)", [30, 60, 90])
     
     #The code below takes past 30 day data using get_data. Then it gets the close
     #price column and uses .pct_change() to get the daily return. Daily volatility 
     #is then calculated as the standard deviation of the daily return.
     @st.cache
     def montecarlo(ticker, time_horizon, simulations):
     
         end_date = datetime.now().date()
         start_date = end_date - timedelta(days=30)
     
         stock_price = si.get_data(ticker, start_date, end_date)
         close_price = stock_price['close']
     
     
         daily_return = close_price.pct_change()
         daily_volatility = np.std(daily_return)
     
         #Initialize the simulation dataframe    
         simulation_df = pd.DataFrame()
     
         for i in range(simulations):        
                      
                # The list to store the next stock price
                next_price = []
    
    #    Create the next stock price
                last_price = close_price[-1]
    
                for x in range(time_horizon):
                               
                      # Generate the random percentage change around the mean (0) and std (daily_volatility)
                      future_return = np.random.normal(0, daily_volatility)

            # Generate the random future price
                      future_price = last_price * (1 + future_return)

            # Save the price and go next
                      next_price.append(future_price)
                      last_price = future_price
    
    #    Store the result of the simulation
                simulation_df[i] = next_price
                
         return simulation_df   
          
#The code below plots the monte carlo simulation using maplotlib. It also calculates
#variance at risk and displays it. the VAR is calculated using the last row of
#the montecarlo simulation. the distribution of this ending price is displaued and
#the 5th percentile of the distribution is marked


     if ticker != '-':
         mc = montecarlo(ticker, time_horizon, simulations)
                  
         end_date = datetime.now().date()
         start_date = end_date - timedelta(days=30)
         
         stock_price = si.get_data(ticker, start_date, end_date)
         close_price = stock_price['close']
         
         fig, ax = plt.subplots(figsize=(15, 10))
         

         ax.plot(mc)
         plt.title('Monte Carlo simulation for ' + str(ticker) + ' stock price in next ' + str(time_horizon) + ' days')
         plt.xlabel('Day')
         plt.ylabel('Price')
         
         
         plt.axhline(y= close_price[-1], color ='red')
         plt.legend(['Current stock price is: ' + str(np.round(close_price[-1], 2))])
         ax.get_legend().legendHandles[0].set_color('red')

         st.pyplot(fig)
         
         # Value at Risk
         st.subheader('Value at Risk (VaR)')
         ending_price = mc.iloc[-1:, :].values[0, ]
         fig1, ax = plt.subplots(figsize=(15, 10))
         ax.hist(ending_price, bins=50)
         plt.axvline(np.percentile(ending_price, 5), color='red', linestyle='--', linewidth=1)
         plt.legend(['5th Percentile of the Future Price: ' + str(np.round(np.percentile(ending_price, 5), 2))])
         plt.title('Distribution of the Ending Price')
         plt.xlabel('Price')
         plt.ylabel('Frequency')
         st.pyplot(fig1)
         
         
         future_price_95ci = np.percentile(ending_price, 5)
         # Value at Risk
         VaR = close_price[-1] - future_price_95ci
         st.write('VaR at 95% confidence interval is: ' + str(np.round(VaR, 2)) + ' USD')
         
         
     
  
#==============================================================================
# Tab 7 Your Portfolio's Trend
#==============================================================================

#The code below uses a multiselect box to allow user to select multiple tickers.
#Then a new dataframe is created with each ticker as a column. A for loop is used to
#populate each column with the close price of that ticker. Then plotly is used to 
#visualize the trend of the selected portfolio
#Reference:
#https://blog.quantinsti.com/stock-market-data-analysis-python/


def tab7():
      st.title("Your Portfolio's Trend")
      alltickers = si.tickers_sp500()
      selected_tickers = st.multiselect("Select tickers in your portfolio", options = alltickers, default = ['AAPL'])
      
      
      df = pd.DataFrame(columns=selected_tickers)
      for ticker in selected_tickers:
          df[ticker] = yf.download(ticker, period = '5Y')['Close']
                
               
      fig = px.line(df)
      st.plotly_chart(fig) 
      
        
    
    
    
#==============================================================================
# Main body
#==============================================================================

def run():
    
    # Add the ticker selection on the sidebar
    # Get the list of stock tickers from S&P500
    ticker_list = ['-'] + si.tickers_sp500()
    
    # Add selection box
    global ticker
    ticker = st.sidebar.selectbox("Select a ticker", ticker_list)
    
    
    # Add a radio box
    select_tab = st.sidebar.radio("Select tab", ['Summary', 'Chart', 'Statistics', 'Financials', 'Analysis', 'Monte Carlo Simulation', "Your Portfolio's Trend"])
    
    # Show the selected tab
    if select_tab == 'Summary':
        tab1()
    elif select_tab == 'Chart':
        tab2()
    elif select_tab == 'Statistics':
        tab3()
    elif select_tab == 'Financials':
        tab4()
    elif select_tab == 'Analysis':
        tab5()
    elif select_tab == 'Monte Carlo Simulation':
        tab6()
    elif select_tab == "Your Portfolio's Trend":
        tab7()
       
    
if __name__ == "__main__":
    run()    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    