# reverse_game_of_life
Predict the state of [Conway's Game of Life](https://en.wikipedia.org/wiki/Conway%27s_Game_of_Life) a few turns ago by implementing [Logistic Regression](https://en.wikipedia.org/wiki/Logistic_regression), [Decision Tree](https://en.wikipedia.org/wiki/Decision_tree), and [Random Forest](https://en.wikipedia.org/wiki/Random_forest) models in Python. (42 Silicon Valley)

## The Problem

This problem was originally posted as a [Kaggle competition](https://www.kaggle.com/c/conway-s-reverse-game-of-life) in 2014.  

Given a 20 x 20 board state and 1 <= `delta` <= 5, predict the board state `delta` turns ago.

## Our Approach

See our [powerpoint presentation](https://docs.google.com/presentation/d/1AWeghT_EnCwOBYbK7sFx-vJ1xOBCPzxUDKLGn8BK21o).

* We train 5 separate models for each `delta` case.
* For each board position, we consider only its neighbors in the surrounding 7 x 7 area.
* We try 3 types of models:
  * `LR` Logistic Regression.
  * `DT` Decision Tree.
  * `RF` Random Forest.
* We compare model performances in cross-validation:
  * Accuracy.
  * [F1 score](https://en.wikipedia.org/wiki/F1_score).

## Prerequisites

You have `python3` and `cython` installed.

## Installing

```
./setup/setup.sh
```
* This also compiles our cython module.

## Running

### Training
```
python3 train.py ( LR | DT | RF ) training_data
```

### Predicting
```
python3 predict.py ( LR | DT | RF ) test_data
```
### Environment Options
Export environment variables:
* `RGOL_CV`=`TRUE` Enable cross-validation.
* `RGOL_VERBOSE`=`TRUE` Enable verbose.
* `RGOL_PLOTS`=`TRUE` Enable plotting (only for Logistic Regression).
