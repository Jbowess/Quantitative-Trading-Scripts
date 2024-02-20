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
logging.basicConfig(filename='forextradebot.log', level=logging.INFO, format='%(asctime)s - %(levelname)s: %(message)s')

# Replace these with your Discord bot token and channel ID
TOKEN = ''
CHANNEL_ID = ''

# List of forex currency pairs
currency_pairs = [
    'USDEUR=X', 'USDJPY=X', 'USDGBP=X', 'USDCHF=X', 'USDCAD=X',  # Major pairs
    'USDHKD=X', 'USDSEK=X', 'USDSGD=X', 'USDNOK=X', 'USDTRY=X',  # Major pairs
    'USDZAR=X', 'USDMXN=X', 'USDKRW=X', 'USDTHB=X', 'USDTWD=X',  # Major pairs
    'USDAUD=X', 'USDCNY=X', 'USDNZD=X', 'USDCNH=X', 'USDINR=X',  # Major pairs
    'USDPHP=X', 'USDSAR=X', 'USDRUB=X', 'USDIDR=X', 'USDMYR=X',  # Major pairs
    'USDPKR=X', 'USDBRL=X', 'USDXAG=X', 'USDXAU=X', 'USDXPD=X',  # Major pairs
    'USDXPT=X', 'USD=X', 'EURUSD=X', 'GBPUSD=X', 'AUDUSD=X',  # Major pairs
    'NZDUSD=X', 'USDJPY=X', 'USDCHF=X', 'USDCAD=X', 'USDHKD=X',  # Major pairs
    'USDSEK=X', 'USDSGD=X', 'USDNOK=X', 'USDTRY=X', 'USDZAR=X',  # Major pairs
    'USDMXN=X', 'USDKRW=X', 'USDTHB=X', 'USDTWD=X', 'USDCNY=X',  # Major pairs
    'USDCNH=X', 'USDRUB=X', 'USDIDR=X', 'USDMYR=X', 'USDPHP=X',  # Major pairs
    'USDSAR=X', 'USDBRL=X', 'USDINR=X', 'USDPKR=X', 'USDXAU=X',  # Major pairs
    'USDXAG=X', 'USDXPD=X', 'USDXPT=X', 'USD=X', 'EURJPY=X',  # Major pairs
    'EURGBP=X', 'EURCHF=X', 'EURCAD=X', 'EURAUD=X', 'EURNZD=X',  # Major pairs
    'EURSEK=X', 'EURDKK=X', 'EURNOK=X', 'EURTRY=X', 'EURZAR=X',  # Major pairs
    'EURHKD=X', 'EURSGD=X', 'EURCNY=X', 'EURINR=X', 'EURIDR=X',  # Major pairs
    'EURMYR=X', 'EURPHP=X', 'EURTHB=X', 'EURPLN=X', 'EURRUB=X',  # Major pairs
    'EURBRL=X', 'EURMXN=X', 'EURARS=X', 'EURCLP=X', 'EURCOP=X',  # Major pairs
    'EURPEN=X', 'EURVEF=X', 'EURAED=X', 'EURSAR=X', 'EURILS=X',  # Major pairs
    'GBPEUR=X', 'GBPJPY=X', 'GBPAUD=X', 'GBPNZD=X', 'GBPSEK=X',  # Major pairs
    'GBPNOK=X', 'GBPTRY=X', 'GBPZAR=X', 'GBPHKD=X', 'GBPSGD=X',  # Major pairs
    'GBPCNY=X', 'GBPINR=X', 'GBPIDR=X', 'GBPMYR=X', 'GBPPHP=X',  # Major pairs
    'GBPTHB=X', 'GBPPLN=X', 'GBPRUB=X', 'GBPDKK=X', 'GBPSEK=X',  # Major pairs
    'GBPISK=X', 'GBPMXN=X', 'GBPCAD=X', 'GBPCHF=X', 'GBPUSD=X',  # Major pairs
    'JPYEUR=X', 'JPYGBP=X', 'JPYAUD=X', 'JPYNZD=X', 'JPYSEK=X',  # Major pairs
    'JPYNOK=X', 'JPYTRY=X', 'JPYZAR=X', 'JPYHKD=X', 'JPYSGD=X',  # Major pairs
    'JPYCNY=X', 'JPYINR=X', 'JPYIDR=X', 'JPYMYR=X', 'JPYPHP=X',  # Major pairs
    'JPYTHB=X', 'JPYPLN=X', 'JPYRUB=X', 'JPYDKK=X', 'JPYSEK=X',  # Major pairs
    'JPYISK=X', 'JPYMXN=X', 'JPYCAD=X', 'JPYCHF=X', 'JPYUSD=X',  # Major pairs
    'AUDJPY=X', 'AUDGBP=X', 'AUDCAD=X', 'AUDNZD=X', 'AUDSEK=X',  # Major pairs
    'AUDNOK=X', 'AUDTRY=X', 'AUDZAR=X', 'AUDHKD=X', 'AUDSGD=X',  # Major pairs
    'AUDCNY=X', 'AUDINR=X', 'AUDIDR=X', 'AUDMYR=X', 'AUDPHP=X',  # Major pairs
    'AUDTHB=X', 'AUDPLN=X', 'AUDRUB=X', 'AUDDKK=X', 'AUDSEK=X',  # Major pairs
    'AUDISK=X', 'AUDMXN=X', 'AUDCAD=X', 'AUDCHF=X', 'AUDUSD=X',  # Major pairs
    'NZDJPY=X', 'NZDEUR=X', 'NZDGBP=X', 'NZDAUD=X', 'NZDSEK=X',  # Major pairs
    'NZDNOK=X', 'NZDTRY=X', 'NZDZAR=X', 'NZDHKD=X', 'NZDSGD=X',  # Major pairs
    'NZDCNY=X', 'NZDINR=X', 'NZDIDR=X', 'NZDMYR=X', 'NZDPHP=X',  # Major pairs
    'NZDTHB=X', 'NZDPLN=X', 'NZDRUB=X', 'NZDDKK=X', 'NZDSEK=X',  # Major pairs
    'NZDISK=X', 'NZDMXN=X', 'NZDCAD=X', 'NZDCHF=X', 'NZDUSD=X',  # Major pairs
]


# Modify the initial_window to cover 10 days
initial_window = 10

# Function to get historical forex data asynchronously
async def get_historical_data_async(currency_pair):
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(f'https://query1.finance.yahoo.com/v7/finance/download/{currency_pair}?period1={int((datetime.now() - timedelta(days=initial_window)).timestamp())}&period2={int(datetime.now().timestamp())}&interval=1d&events=history') as response:
                data = await response.text()
                prices = [float(line.split(',')[4]) for line in data.split('\n')[1:] if line]
                logging.info(f'Successfully pulled historical data for {currency_pair}')
                return prices
    except Exception as e:
        logging.error(f'Error fetching historical data for {currency_pair}: {e}')
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

async def generate_trading_signal(currency_pair):
    try:
        prices = await get_historical_data_async(currency_pair)
        if len(prices) < 2:
            logging.info(f'Insufficient data for {currency_pair}')
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
        
        # Trading strategy based on position
        if position in ['Below Buy Threshold', 'Above Sell Threshold']:
            # Current price is the latest price in the historical data
            current_price = prices[-1]
            
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
                f'**{position_text} SIGNAL : {currency_pair}**\n'
                f'{entry_emoji} Entry Price: {current_price:.2f}\n'
                f'{stop_loss_emoji} Stop Loss: {stop_loss:.2f}\n'
                f'{take_profit_emoji} Take Profit 1: {take_profit_1:.2f}\n'
                f'{take_profit_emoji} Take Profit 2: {take_profit_2:.2f}\n'
                f'{take_profit_emoji} Take Profit 3: {take_profit_3:.2f}\n'
                f'{success_ratio_emoji} AI Driven Success Ratio: {success_ratio:.2f}\n'
                f'{bid_recommendation_emoji} AI Bid Recommendation: {bid_recommendation}\n'
                f'{"-" * 40}\n'
            )

            logging.info(f'Successfully generated {"buy" if position == "Below Buy Threshold" else "sell"} signal for {currency_pair}')
            return signal
        else:
            logging.info(f'No signal generated for {currency_pair}')
            return None  # No signal
        
    except Exception as e:
        logging.error(f'Error generating trading signal for {currency_pair}: {e}')
        return f'Error generating trading signal for {currency_pair}'



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
        tasks = [generate_trading_signal(currency_pair) for currency_pair in currency_pairs]
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
