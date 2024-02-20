import asyncio
import aiohttp
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import logging
import discord
from discord.ext import tasks

intents = discord.Intents.default()
client_discord = discord.Client(intents=intents)

# Initialize logging
logging.basicConfig(filename='stockmarkettradebot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Replace these with your Discord bot token and channel ID
TOKEN = ''
CHANNEL_ID = ''

# List of ASX Top 200
stock_symbols = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'FB',  # Top technology stocks
    'TSLA', 'NVDA', 'PYPL', 'INTC', 'ADBE',  # Top tech and growth stocks
    'JPM', 'BAC', 'WFC', 'C', 'GS',  # Top bank stocks
    'KO', 'PEP', 'MCD', 'SBUX', 'NKE',  # Top consumer goods stocks
    'XOM', 'CVX', 'BP', 'RDS-A', 'TOT',  # Top energy stocks
    'DIS', 'NFLX', 'CMCSA', 'T', 'VZ',  # Top entertainment and telecom stocks
    'CRM', 'ORCL', 'IBM', 'CSCO', 'INTU',  # Top software and tech stocks
    'PG', 'UNH', 'PFE', 'MRK', 'JNJ',  # Top healthcare and pharmaceuticals
    'HD', 'LOW', 'AMT', 'CCI', 'PLD',  # Top real estate and construction
    'CVS', 'WBA', 'LLY', 'ABT', 'MDT',  # Top healthcare and pharmaceuticals
    'BA', 'LMT', 'RTX', 'GD', 'NOC',  # Top aerospace and defense stocks
    'MS', 'GS', 'V', 'MA', 'AXP', 'SQ',  # Top financial services and payment stocks
    'UNP', 'CSX', 'NSC', 'FDX', 'UPS',  # Top transportation and logistics stocks
    'GOOG', 'BABA', 'TCEHY', 'BIDU',  # Top technology and e-commerce stocks
    'BRK-A', 'BRK-B', 'WMT',  # Top conglomerates and retail stocks
    'PM', 'MO',  # Top tobacco stocks
    'QCOM', 'MMM', 'CAT', 'GE', 'HON',  # Top industrial and manufacturing stocks
    'DPZ', 'CMG', 'YUM', 'MCD', 'SBUX',  # Top restaurant and food stocks
    'SBAC', 'EQIX',  # Top real estate and REIT stocks
    'ABBV', 'ANTM', 'CI', 'HCA', 'HUM',  # Top healthcare and insurance stocks
    'AMD', 'BMY',  # Additional pharmaceutical stocks
]


# Modify the initial_window to cover 10 days
initial_window = 10

# Function to get historical forex data asynchronously
async def get_historical_data_async(stock_symbols):
    try:
        stock_data = yf.download(stock_symbols, start=(datetime.now() - timedelta(days=initial_window)), end=datetime.now())
        prices = stock_data['Close'].values
        logging.info(f'Successfully pulled historical data for {stock_symbols}')
        return prices
    except Exception as e:
        logging.error(f'Error fetching historical data for {stock_symbols}: {e}')
        return []

# Function to calculate moving averages
def calculate_moving_average(prices, window):
    return np.convolve(prices, np.ones(window), 'valid') / window

# Function to calculate RSI (Relative Strength Index)
def calculate_rsi(prices, window=14):
    delta = np.diff(prices)
    gain = delta * (delta > 0)
    loss = -delta * (delta < 0)
    avg_gain = calculate_moving_average(gain, window)
    avg_loss = calculate_moving_average(loss, window)
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_success_ratio(prices):
    """
    Calculate the success percentage based on historical prices.
    """
    if len(prices) < 2:
        return None  # Insufficient data
    
    previous_price = prices[-2]
    current_price = prices[-1]
    
    if previous_price == 0:
        return None  # Avoid division by zero
    
    success_ratio = ((current_price - previous_price) / previous_price) * 100
    return success_ratio

def calculate_bid_recommendation(success_ratio):
    """
    Determine bid recommendation based on success percentage.
    """
    if success_ratio > 1.5:
        return 'High'
    elif success_ratio < 0.8:
        return 'Low'
    else:
        return 'Medium'

async def generate_trading_signal(stock_symbol):
    try:
        prices = await get_historical_data_async(stock_symbol)
        if len(prices) < 2:
            logging.info(f'Insufficient data for {stock_symbol}')
            return None  # No signal
        
        # Calculate RSI to determine signal
        rsi = calculate_rsi(prices)
        current_rsi = rsi[-1]

        # Trading strategy based on RSI
        if current_rsi < 30:
            position = 'Below Buy Threshold'
        elif current_rsi > 70:
            position = 'Above Sell Threshold'
        else:
            position = 'Indeterminate'
        
        # Current price is the latest price in the historical data
        current_price = prices[-1]
        
        # Trading strategy based on position
        if position in ['Below Buy Threshold', 'Above Sell Threshold']:
            # Calculate entry price
            entry_price = current_price
            
            # Calculate success percentage and bid recommendation
            success_ratio = calculate_success_ratio(prices)
            bid_recommendation = calculate_bid_recommendation(success_ratio)


            # Adjust stop loss and take profit levels based on bid recommendation
            if bid_recommendation == 'High':
                if position == 'Below Buy Threshold':  # For long position
                    stop_loss = current_price * 0.997  # 0.3% below entry price
                    take_profit_1 = current_price * 1.03  # 0.3% above entry price
                    take_profit_2 = current_price * 1.05  # 0.5% above entry price
                    take_profit_3 = current_price * 1.07  # 0.7% above entry price
                else:  # For short position
                    stop_loss = current_price * 1.003  # 0.3% above entry price
                    take_profit_1 = current_price * 0.97  # 0.3% below entry price
                    take_profit_2 = current_price * 0.95  # 0.5% below entry price
                    take_profit_3 = current_price * 0.93  # 0.7% below entry price
            elif bid_recommendation == 'Medium':
                if position == 'Below Buy Threshold':  # For long position
                    stop_loss = current_price * 0.996  # 0.4% below entry price
                    take_profit_1 = current_price * 1.02  # 0.2% above entry price
                    take_profit_2 = current_price * 1.04  # 0.4% above entry price
                    take_profit_3 = current_price * 1.06  # 0.6% above entry price
                else:  # For short position
                    stop_loss = current_price * 1.004  # 0.4% above entry price
                    take_profit_1 = current_price * 0.98  # 0.2% below entry price
                    take_profit_2 = current_price * 0.96  # 0.4% below entry price
                    take_profit_3 = current_price * 0.94  # 0.6% below entry price
            elif bid_recommendation == 'Low':
                if position == 'Below Buy Threshold':  # For long position
                    stop_loss = current_price * 0.995  # 0.5% below entry price
                    take_profit_1 = current_price * 1.01  # 0.1% above entry price
                    take_profit_2 = current_price * 1.02  # 0.2% above entry price
                    take_profit_3 = current_price * 1.03  # 0.3% above entry price
                else:  # For short position
                    stop_loss = current_price * 1.005  # 0.5% above entry price
                    take_profit_1 = current_price * 0.99  # 0.1% below entry price
                    take_profit_2 = current_price * 0.98  # 0.2% below entry price
                    take_profit_3 = current_price * 0.97  # 0.3% below entry price

            # Define emojis
            entry_emoji = '🪙'
            stop_loss_emoji = '🚨'
            take_profit_emoji = '💰'
            success_ratio_emoji = '📈' if success_ratio > 0 else '📉'
            bid_recommendation_emoji = '🔼' if bid_recommendation == 'High' else '➡️' if bid_recommendation == 'Medium' else '🔽'
            
            # Define position text and emoji
            position_text = 'LONG 📈' if position == "Below Buy Threshold" else 'SHORT 📉'

            # Construct the signal string with emojis and trade information
            signal = (
                f'**{position_text} SIGNAL : {stock_symbol}**\n'
                f'{entry_emoji} Entry Price: {entry_price:.2f}\n'
                f'{stop_loss_emoji} Stop Loss: {stop_loss:.2f}\n'
                f'{take_profit_emoji} Take Profit 1: {take_profit_1:.2f}\n'
                f'{take_profit_emoji} Take Profit 2: {take_profit_2:.2f}\n'
                f'{take_profit_emoji} Take Profit 3: {take_profit_3:.2f}\n'
                f'{success_ratio_emoji} AI Driven Success Ratio: {success_ratio:.2f}\n'
                f'{bid_recommendation_emoji} AI Bid Recommendation: {bid_recommendation}\n'
                f'{"-" * 40}\n'
            )

            logging.info(f'Successfully generated {"buy" if position == "Below Buy Threshold" else "sell"} signal for {stock_symbol}')
            return signal
        else:
            logging.info(f'No signal generated for {stock_symbol}')
            return None  # No signal
        
    except Exception as e:
        logging.error(f'Error generating trading signal for {stock_symbol}: {e}')
        return f'Error generating trading signal for {stock_symbol}'


# Function to send message to Discord channel
async def send_message(message):
    try:
        channel = client_discord.get_channel(int(CHANNEL_ID))
        if channel is not None:
            await channel.send(message)
            logging.info(f'Successfully sent message: {message}')
        else:
            logging.error(f'Error sending message: Channel not found')
    except Exception as e:
        logging.error(f'Error sending message: {e}')

# Function to handle trading signals
@tasks.loop(minutes=1)
async def handle_signals():
    try:
        tasks = [generate_trading_signal(stock_symbol) for stock_symbol in stock_symbols]
        results = await asyncio.gather(*tasks)
        
        # Filter successful signals
        successful_signals = [result for result in results if result is not None]
        
        # Sort signals by success percentage
        successful_signals.sort(key=lambda x: float(x.split('\n')[1].split(': ')[1][:-1]), reverse=True)
        
        # Send the top 3 trade signals to Discord
        if successful_signals:
            top_3_signals = successful_signals[:3]
            for signal in top_3_signals:
                await send_message(signal)
            
    except Exception as e:
        logging.error(f'Error handling signals: {e}')

# Discord event: on ready
@client_discord.event
async def on_ready():
    try:
        print(f'Logged in as {client_discord.user}')
        handle_signals.start()
    except Exception as e:
        logging.error(f'Error on ready: {e}')

# Run the Discord bot
if __name__ == '__main__':
    try:
        client_discord.run(TOKEN)
    except Exception as e:
        logging.error(f'Error running Discord bot: {e}')
