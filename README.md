# TradingRLBot

# Table of contents
- [TradingRLBot](#tradingrlbot)
- [Table of contents](#table-of-contents)
- [Introduction and motivation](#introduction-and-motivation)
- [Dataset](#dataset)
  * [Data](#data)
  * [Preprocessing and feature engineering](#preprocessing-and-feature-engineering)
- [Environment](#environment)
  * [Buy and Sell](#buy-and-sell)
    + [Action mapping](#action-mapping)
  * [Updating the Portfolio](#updating-the-portfolio)
  * [Updating the state](#updating-the-state)
  * [Computing the reward](#computing-the-reward)
- [Models](#models)
  * [DQN](#dqn)
  * [DDPG](#ddpg)
  * [PPO](#ppo)
- [Evaluation](#evaluation)
  * [Metrics](#metrics)
  * [Training and Testing](#training-and-testing)
  * [Time series validation](#time-series-validation)
- [Hyperparameter tuning](#hyperparameter-tuning)
- [Experiments](#experiments)
  * [Hypothesis](#hypothesis)
  * [Experiment setup](#experiment-setup)
  * [Results](#results)
    + [Best models obtained using Time Series Validation](#best-models-obtained-using-time-series-validation)
      - [From 2020-02-01 to 2021-04-01](#from-2020-02-01-to-2021-04-01)
    + [Testing results - Period 1](#testing-results---period-1)
    + [Testing results - Period 2](#testing-results---period-2)
    + [Allocations along time - Period 2](#allocations-along-time---period-2)
    + [Transactons statistics - Period 2](#transactons-statistics---period-2)
  * [Conclusions](#conclusions)
- [How to run the code](#how-to-run-the-code)
  * [Set up the Conda environment](#set-up-the-conda-environment)
  * [How to run the hyperparameter tuning](#how-to-run-the-hyperparameter-tuning)
  * [How to visualize the results in mlflow](#how-to-visualize-the-results-in-mlflow)
  * [How to visualize the results in tensorboard](#how-to-visualize-the-results-in-tensorboard)
- [References](#references)



# Introduction and motivation
2021 was a crazy year for Bitcoin and the cryptocurrency world in general. Bitcoin reached its all-time high of 64'000 USD and a lot of new exciting projects related to blockchain were developed. Several people start using these coins and trade them, attracted by the incredible rise in price. We decided then to test if the powerful neural networks could help us trading these coins. 

The idea of the project is to make a reinforcement learning agent trade eight different coins for several months using just an initial budget in cash. As for the humans performing technical analysis, the agent has access just to some historical data of the price and volume. 


---

# Dataset
## Data
The data that we used to build our dataset was downloaded from [Binance](https://www.binance.com/), the biggest bitcoin exchange and altcoin crypto exchange in the world. The dataset provides the history of **hourly** prices in **USD** for the **top 8 cryptocurrencies** of the market ("BTC", "ETH", "BNB", "ADA", "XRP", "DOGE", "LINK", "LTC") and starts at **2020-01-01 00:00:00** and ends at **2021-06-30 23:00:00**.

![](https://i.imgur.com/jXGoZBA.jpg)

## Preprocessing and feature engineering
In order to feed to the network clean and meaningful data we have performed the following operations:
1. Fill **missing values**. We have filled the missing values uding the forward fill method, which propagates the last valid observation forward.
2. Add **technical indicators**. These indicators are pattern-based signals produced by the variation in price and volume. We have divided these indicators into two types:
    - **Short term indicators** To describe the variation of the price and value during the same day. We consider all the indicators offered by the library ta (over 90) and we selected the less correlated ones (cor <0.7) in the first month of data. We were left with 18 indicators.
    - **Long term indicators** To describe what happened in the previous month we use four famous indicators: smooth moving average, directional moving index, relative strength index and commodity channel index. We chose three different windows: 1 day, 7 days and 30 days. In this way, we obtained other 12 technical indicators
3. Add **percentage change** of the variables. To help the network to recognise the patterns, even if the absolute values are changing in time, we have calculated the daily change in percentage of price, volumes and all the technical indicators.
4. Add **covariance between the coins** price. We have calculated the covariance among the different coins' closing prices, in order to provide information about the relationship among these cryptocurrencies.
5. Eliminate **the first 30 days** of data. We couldnot consider the first 30 days since our metrics need a 30 days window period.

![](https://i.imgur.com/82vkIxB.jpg)


---

# Environment

Our agents are trained in an environment developled by ourselves which emulates the trading options available in a crypto exchange.

![](https://i.imgur.com/aSa4TcO.png)




## Buy and Sell

The output of the agent is mapped to the environment actions that decide whether to **Buy**, **Sell** or **Hold** stock.

* **Buy**: If the agent has enough cash in the **Portfolio**, perform a buy. This means adding the bought assets to the portfolio while substracting their value and the comissions from the cash.
* **Sell**: If the agent has enough assets in the **Portfolio**, perform a sell. This means adding the value of the sold assets to the cash while substracting the amount of assets sold from the portfolio.
* **Hold**: Do nothing.

All actions are limited by a parameter called *max_amount_per_trade*, defined when creating the environment. This parameter helps to control that the algorithm does not trade with the whole portfolio value in each operation.

### Action mapping

We need then to map the action performed by the agent to the environment action. This map is different depending if the output of the agent is discrete or continuous:

* Continuous actions: The action space is defined as a tensor of shape (1, *number of assets*), with values ranging from -1 to 1. The values are then multiplied by the *max_amount_per_trade* parameter.
In example, havingthe actions of an environment with just three assets were represented as:
```
# With max_amount_per_trade = 1000
actions = [0.5, -0.1, 1]

# The actions would be mapped to:
# - Buy 50$ of the first asset
# - Sell 10$ of the second asset
# - Buy 100$ of the third asset
```
* Discrete actions: The output of the models is defined as a integer ranging from [0, 2*number of assets]. A value of 0 means holding. Odd numbers are buy operations and even numbers are sell ones. In this case, each operation trades with the whole *max_amount_per_trade*.

![](https://i.imgur.com/HLsGSYP.png)

## Updating the Portfolio

The **Portfolio** object holds the trading logic and keeps track of how the capital is allocated across the different assets at each point. At the beggining of an experiment everything is in cash, but with every buy and sell the allocations are modified. 

At the end of the experiment, the capital that the Portfolio is holding can be compared to the one in the initial stat to measure the return of the startegy. It can also show when and which the transactions were done.

## Updating the state

The state in each timestep is defined bya vector composed of:
* The amount of cash in the portfolio
* The amount of the assets in the portfolio
* The price of assets (close values)
* The features built for all assets in the preprocessing step

In every new timestep we need to update all this values since the market conditions and the portfolio allocations might have changed.

## Computing the reward

Agents are rewarded at every timestep with the difference between the value of the portfolio before and after performing the actions. At the beginning we were using the difference in absolute value (portfolio_after - portfolio_before), but then we decided to use the change in percentage ((portfolio_after - portfolio_before)/portfolio_before )). With the second approach we obtained as expected better results, since the return in percentage is independent of the size of the portfolio which can change a lot in different time period.

Other reward functions have been implemented (Sharpe, sortino) but there was no time to test them. These functions would have been more aligned with our trading strategy. More information in the *Metrics* section.

---
# Models
We have tried three different models, one uses the discete action space previously defined (DQN) and the other two the continuous action space (DDPG and PPO).
## DQN
The **deep Q-Network** is a off-policy Q-learning which use different tricks to stabilize the learning like a replay buffer and a target network. To have a nice introduction to Reinforment learning and DQN you can look at the following links: [a-long-peek-into-reinforcement-learning](https://lilianweng.github.io/lil-log/2018/02/19/a-long-peek-into-reinforcement-learning.html). For the network we have left the default architecture proposed by the paper [Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236): two hidden layers with 64 neurons each, RELUs for all hidden layers and a tanh activation in the last layer. The optimizer used is ADAM. 
## DDPG
The **deep deterministic policy gradient** is a model free, off-policy algorithm which combines DPG and DQN. To know more about the policy gradient algorithms you can look at this  blog: [policy-gradient-algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html). For all the networks involved we have left the default architecture proposed by the paper [Continuous control with deep reinforcement learning](https://arxiv.org/abs/1509.02971v6): two hidden layers of respecively 400 and 300 neurons, ReLUs for all hidden layers and the final output layer of the actor was a tanh layer, to bound the actions. The optimizer used is ADAM. 

## PPO

The **proximal policy optimization** is a model free, on-policy algorithm which uses the actor-critic framework. It combines ideas from A2C (having multiple workers) and TRPO (it uses a trust region to improve the actor). If you want to know more you can look again at this blog: [policy-gradient-algorithms](https://lilianweng.github.io/lil-log/2018/04/08/policy-gradient-algorithms.html). We have used the architecture proposed by the [stable baseline](https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html) library. For all the networks we have: two hidden layers with 64 neurons each, RELUs for all hidden layers and a tanh activation in the last layer. The optimizer used is ADAM. 

---
# Evaluation

## Metrics

The metrics used to evaluate the strategy of the agents are commonly used in the finance world. Each of them has its own benefits and drawbacks:

* **Return**: The relative difference between the final portfolio value and the initial one.
* **Sharpe ratio**: Similar to return but considering the volatility of the returns. A higher volatily means a lower sharpe, which also means a less desirable startegy. [(More)](https://www.investopedia.com/terms/s/sharperatio.asp)
* **Sortino ratio**: Almost identical to sharpe, but in this case only the negative volatily is penalizing the final value.[(More)](https://www.investopedia.com/terms/s/sortinoratio.asp#:~:text=What%20Is%20the%20Sortino%20Ratio,standard%20deviation%20of%20portfolio%20returns.)
* **Max drawdown**: Maximum observed loss from a peak to a trough of a portfolio, before a new peak is attained.[(More)](https://www.investopedia.com/terms/m/maximum-drawdown-mdd.asp)

## Training and Testing

Testing steps evaluate the performance of a model trained with data from a previous period. It's important not to test the model with data prior to the training set, as it would result in data leaking. 

When tuning models, tests are not recommended since the results of different experiments mught be overfitted to the test set.

## Time series validation

In a TSV, a number N of training-testing sets are ran and their results are averaged. This approach is far more robust than a simple testing and allows obtaining better results when tuning models.

![](https://i.imgur.com/CKTnveo.png)


As it can be seen in the figure, there's an intentional gap between the training and the testing set. This is recommended in the industry to make sure that the two sets are more independent.

---
# Hyperparameter tuning
In order to obtain the best possible results, an exhaustive hyperparameter tuning was made.
To do this, we used [Optuna](https://optuna.org/), an open source hyperparameter optimization framework to automate hyperparameter search.
Since we needed to optimize three different algorithms (DQN, PPO, DDPG) with differents hyperparameters each one, we decided to set up three independent configurations to apply the hyperparameter tuning. We used a TPE (Tree-structured Parzen Estimator) algorithm and saved all the trials into a MySQL as a backup database and also with MLflow.

For **PPO**:

| Parameter| Range    | Best configuration    |
| -------- | -------- | -------- |
| n_steps     | categorical([100, 500, 1000, 2000, 3000])     | 1000     |
| ent_coef     | loguniform(0.01, 0.1)     | 0.0108486     |
| learning_rate     | loguniform( 1e-3, 1e-2)     | 0.00637956    |
| batch_size     | categorical([2, 5, 10, 20, 50, 100])      | 20    |
| n_epochs     | categorical([3, 5, 10])     | 3     |
| gamma     | categorical([0.99, 0.995, 0.999, 0.9999])     | 0.995     |
| gae_lambda     | loguniform(0.9, 0.99999)     | 0.969964    |


For **DQN**:

| Parameter| Range    | Best configuration      |
| -------- | -------- | -------- |
| batch_size     | discrete_uniform(32, 256, 2)| 100     |
| gamma     | loguniform(0.9, 0.99999)     | 0.965     |
| learning_rate     | loguniform(0.0005, 0.01)     | 0.002     |
| tau     | discrete_uniform(32, 256, 2)     | 0.014     |


For **DDPG**:

| Parameter| Range    | Best configuration      |
| -------- | -------- | -------- |
| gamma     | loguniform(0.9, 0.99)     | 0.975     |
| tau     | uniform(0.001, 0.1)     | 0.0126     |
| learning_rate     | loguniform(0.001, 0.01)     | 0.00297     |
| batch_size     | categorical([32, 128, 1])     | 128     |
| buffer_size     | categorical([100000, 1000000, 10000] )     | 10000     |



---
# Experiments 
![](https://i.imgur.com/XNLXp10.png)

## Hypothesis
The main hypothesis of the project is to prove if a **reinforcement learning agent** can develop **profitable** strategies in a cryptocurrency environment.
## Experiment setup
To test our hypothesis we have build the following experimental setup:
1. Download, create, clean and augment a cryptocurrency **dataset**.
2. Create a custom gym **environment** with the previous dataset that emulates the trading options available in a crypto exchange.
3. Implement three reinforcement learning **algorithms** (DQN, PPO, DDPG).
4. Apply **hyperparameter tuning** and **time series validation** to get the best hyperparameter configuration for each model.
5. **Train** and **test** each model with its best hyperparameter configuration.

## Results

### Best models obtained using Time Series Validation
#### From 2020-02-01 to 2021-04-01

| Model | return | sharpe | sortino | max\_drawdown | return\_btc | sharpe\_btc |
| ----- | ------ | ------ | ------- | ------------- | ----------- | ----------- |
| PPO   | 0.912  | \-3.79 | \-5.81  | \-0.434       | 1.034       | 9.575       |
| DQN   | 1.004  | 4.828  | 8.541   | \-0.105       | 1.034       | 9.575       |
| DDPG  | 1.415  | 25.396 | 35.857  | \-0.218       | 1.034       | 9.575       |

### Testing results - Period 1
**Model strategies + Bitcoin**
![](https://i.imgur.com/38t7fKT.png)

**Model strategies + All assets**
![](https://i.imgur.com/p9W1IH2.png)

**Summary**

All of the strategies are outperforming Bitcoin in this timeframe.

![](https://i.imgur.com/He9R22r.png)

### Testing results - Period 2
**Model strategies + Bitcoin**
![](https://i.imgur.com/Lff8Btm.png)

**Model strategies + All assets**
![](https://i.imgur.com/3nICKHB.png)

**Summary**
None of the strategies are outperforming Bitcoin in this timeframe.

![](https://i.imgur.com/nPDtlGv.png)

### Allocations along time - Period 2
![](https://i.imgur.com/XR3dTVL.png)
![](https://i.imgur.com/PkXZXr6.png)
![](https://i.imgur.com/xvYycVg.png)

### Transactons statistics - Period 2
| Model | Number of transactions | Average transaction value |
| ----- | ---------------------- | ------------------------- |
| PPO   | 897                    | 315.26                    |
| DQN   | 69                     | 384.06                    |
| DDPG  | 614                    | 335.10                    |

The discrete model is performing less transactions than the continuous ones. This allows it to waste less money in transaction commissions.

## Conclusions
Analyzing the previous results we obtained the following conclusions:

* DQN and DDPG achieve positives returns but PPO doesn't. DDPG outperforms DQN and PPO in the time series validation since it has a higher return. The DDPG's return also outperforms BTC's return in the time series validation  
* For the test-period 1 the three models achieve a return > 1, meaning that they are profitable! DQN is the best model in this test with a return of 1.9, and the three algorithms outperform BTC's return since it is < 1. Another important metric is the max_drawdown which is around [-0.55,-0.4] meaning that the three algorithms almost lose half of their budget.
* For the test-period 2 the three models achieve a return < 1 meaning that they are not profitable (the entire market goes down). Although all the models perform poorly in this second test, DQN it's the one that has the higher return, we think that this fact occurs because DQN has a discrete action space and it learns properly to hold. This reasoning is also supported by the number of transactions made by each algorithm. We can see that DQN has only 69 transactions during the period.
* Finally, we also observe that DQN and PPO are not capable to distribute the budget along with the multiple coins to minimize risk and they focus too much in bnb.

---
# How to run the code
## Set up the Conda environment
1. Clone the repository `git clone https://github.com/Toroi01/TradingRLBot.git`
2. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html)
3. Open a terminal located in the root of the repo and type `conda create -n TradingRLBot python=3.9` to create the environment.
4. Activate the environment typing `conda activate TradingRLBot`
5. Install the requirements typing `pip install -r requirements.txt`

## How to run the hyperparameter tuning
If you want to test different hyperparameters for the model **PPO**. These are the steps:

1. *(Optional)* Go to the python file `.\src\scripts\hyp_param_tuning_PPO.py` and change the following parameter accordingly to your needs:
    *numsplits:* number of different sets used in the time series validation.
    *total_timesteps_model*: number of timesteps considered in every trial.
    *n_trials*: number of  different combinations of hyperparameters to test.
3. *(Optional)* Go to the file `.\src\hyperparameter_tuning\ppo_tune.py` and change the hyperparameters to test in the function `sample_ppo_params` accordingly to your needs.
4. Exectute the script typing `python .\src\scripts\hyp_param_tuning_PPO.py`
5. *(Output)* A folder name \$TIMESTAMP\$ _ppo will appear inside the folder `.\logs\hyptune` for every trial *k* the following three pickle files will be produced:
   *trial_k_HYP.pkl* the hyperparamter tried in that trial
   *trial_k_METRICS.pkl* the metrics obtained in the test set
   *trial_k_MODEL.pkl* the model obtained during the training
   

*NOTE* To try the other two models you need to consider the following:
**DQN** script to run: `.\src\scripts\hyp_param_tuning_DQN.py` ;  file containing the hyperparameter: `.\src\hyperparameter_tuning\dqn_tune.py`
**DDPG** script to run `.\src\scripts\hyp_param_tuning_DDPG.py` ; file containing the hyperparameter: `.\src\hyperparameter_tuning\ddpg_tune.py`

## How to visualize the results in mlflow
You can nicely explore the results obtained with the different hyperpatameters tuning trials in the follwing way:
1.  Run a hyperparameters tuning script as explained previously.
2.  Type `mlflow ui`
3.  Open the url that appear in the terminal 
4.  (Output) For every trial you can check all the hyperparameres that have been tested and the results obtained (return, sharpe, sortino, etc.).

## How to visualize the results in tensorboard
You can also get some more in deepth information about the training in the following way:

1. Run a hyperparameters tuning script as explained previously.
2. Type `tensorboard --logdir logs/`
3. Open the url that appear in the terminal.
4. (Output) In the webpage of tensorbord you can visualize different metrics (divided in scalars and time series) for the different trials and model run.

# References
## Code
[Stable Baselines 3](https://stable-baselines3.readthedocs.io/en/master/)

[Optuna tuner](https://optuna.org/)

[FinRL Library](http://finrl.org/)

[OpenAI](https://openai.com/)

## Algorithms
[PPO](https://arxiv.org/abs/1707.06347)

[DQN](https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf)

[DDPG](https://arxiv.org/abs/1509.02971)

## Finance
[Investopedia](https://www.investopedia.com/)

[Binance](https://www.binance.com/es)
