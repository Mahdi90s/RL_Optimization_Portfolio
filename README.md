# Reinforcement Learning for Portfolio Optimization

## Project Overview
This project explores portfolio optimization using reinforcement learning (RL) techniques.  
The main goal is to build a hierarchical RL agent that selects from multiple advanced algorithms to optimize asset allocation dynamically.  
The project also compares RL methods against classical portfolio optimization approaches like Mean-Variance Optimization and Equal-Weighted Portfolios.

## Data
- Historical daily price data for 10 assets (e.g., AAPL, MSFT, NVDA, AMZN, JPM, etc.) spanning the last 5 years.  
- Initial focus on two assets (AAPL and MSFT) for prototyping and validation.  
- Data stored in `/data/raw/` as CSV files.

## Project Structure
- `/data/raw/`: Raw CSV datasets.  
- `/src/`: Python scripts for data processing, modeling, and evaluation.  
- `/notebooks/`: Jupyter notebooks for exploration and prototyping.  
- `/reports/`: Analysis reports and visualization outputs.

## Technologies Used
- Python (pandas, numpy, yfinance, gym, stable-baselines3, etc.)  
- SQL Server (planned for data storage and management)  
- Power BI (planned for data visualization)

## Next Steps
- Feature engineering and normalization  
- Development of multiple RL algorithms and hierarchical agent design  
- Benchmarking against classical optimization methods  
- Hyperparameter tuning and market regime integration

---

*This project is part of an MSc Data Science dissertation at the University of Essex.*

