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
logging.basicConfig(filename='cryptomarketbot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Replace these with your Discord bot token and channel ID
TOKEN = ''
CHANNEL_ID = ''

# List of crypto Top 100
crypto_symbols = ['BTC-USD', 'ETH-USD', 'BNB-USD', 'USDT-USD', 'SOL-USD',  # Top cryptocurrencies
    'ADA-USD', 'XRP-USD', 'DOT-USD', 'DOGE-USD', 'AVAX-USD',  # Top cryptocurrencies
    'LUNA-USD', 'SOL-USD', 'UNI3-USD', 'LINK-USD', 'LTC-USD',  # Top cryptocurrencies
    'ALGO-USD', 'ATOM-USD', 'BCH-USD', 'WBTC-USD', 'MATIC-USD',  # Top cryptocurrencies
    'ICP-USD', 'FIL-USD', 'AXS-USD', 'TRX-USD', 'XLM-USD',  # Top cryptocurrencies
    'VET-USD', 'EGLD-USD', 'FTT-USD', 'AAVE-USD', 'XTZ-USD',  # Top cryptocurrencies
    'HT-USD', 'EOS-USD', 'RUNE-USD', 'ETC-USD', 'CAKE-USD',  # Top cryptocurrencies
    'SHIB-USD', 'DASH-USD', 'NEO-USD', 'MKR-USD', 'THETA-USD',  # Top cryptocurrencies
    'XMR-USD', 'WAVES-USD', 'CHZ-USD', 'EGLD-USD', 'HOT-USD',  # Top cryptocurrencies
    'ZEC-USD', 'SC-USD', 'SNX-USD', 'GRT-USD', 'IOST-USD',  # Top cryptocurrencies
    'KSM-USD', 'ICX-USD', 'BTT-USD', 'COMP-USD', 'OMG-USD',  # Top cryptocurrencies
    'DCR-USD', 'QTUM-USD', 'ANKR-USD', 'ZIL-USD', 'RVN-USD',  # Top cryptocurrencies
    'YFI-USD', 'CELO-USD', 'FTM-USD', 'RUNE-USD', 'ENJ-USD',  # Top cryptocurrencies
    'DGB-USD', 'SUSHI-USD', 'ONT-USD', 'MANA-USD', 'CRV-USD',  # Top cryptocurrencies
    'HNT-USD', 'STX-USD', 'BNT-USD', 'LRC-USD', 'DENT-USD',  # Top cryptocurrencies
    'WRX-USD', 'HBAR-USD', 'ZEN-USD', 'LSK-USD', 'KCS-USD',  # Top cryptocurrencies
    'ICP-USD', 'BTG-USD', 'ALICE-USD', 'SAND-USD', 'BTS-USD',  # Top cryptocurrencies
    'LPT-USD', 'CELR-USD', 'REP-USD', 'LINA-USD', 'AR-USD',  # Top cryptocurrencies
    'SNT-USD', 'LINA-USD', 'CKB-USD', 'PERP-USD', 'LRC-USD',  # Top cryptocurrencies
]

# Modify the initial_window to cover 10 days
initial_window = 10

# Function to get historical forex data asynchronously
async def get_historical_data_async(crypto_symbols):
    try:
        start_date = datetime.now() - timedelta(days=initial_window)
        end_date = datetime.now()
        # print(f"Fetching historical data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        stock_data = yf.download(crypto_symbols, start=start_date, end=end_date)        
        # print(stock_data)  # Print the retrieved data
        prices = stock_data['Close'].values
        logging.info(f'Successfully pulled historical data for {crypto_symbols}')
        return prices
    except Exception as e:
        logging.error(f'Error fetching historical data for {crypto_symbols}: {e}')
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

async def generate_trading_signal(crypto_symbol):
    try:
        prices = await get_historical_data_async(crypto_symbol)
        if len(prices) < 2:
            logging.info(f'Insufficient data for {crypto_symbol}')
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
                f'**{position_text} SIGNAL : {crypto_symbol}**\n'
                f'{entry_emoji} Entry Price: {entry_price:.2f}\n'
                f'{stop_loss_emoji} Stop Loss: {stop_loss:.2f}\n'
                f'{take_profit_emoji} Take Profit 1: {take_profit_1:.2f}\n'
                f'{take_profit_emoji} Take Profit 2: {take_profit_2:.2f}\n'
                f'{take_profit_emoji} Take Profit 3: {take_profit_3:.2f}\n'
                f'{success_ratio_emoji} AI Driven Success Ratio: {success_ratio:.2f}\n'
                f'{bid_recommendation_emoji} AI Bid Recommendation: {bid_recommendation}\n'
                f'{"-" * 40}\n'
            )

            logging.info(f'Successfully generated {"buy" if position == "Below Buy Threshold" else "sell"} signal for {crypto_symbol}')
            return signal
        else:
            logging.info(f'No signal generated for {crypto_symbol}')
            return None  # No signal
        
    except Exception as e:
        logging.error(f'Error generating trading signal for {crypto_symbol}: {e}')
        return f'Error generating trading signal for {crypto_symbol}'



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
        tasks = [generate_trading_signal(crypto_symbol) for crypto_symbol in crypto_symbols]
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
