import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import copy 
import pandas as pd
from scipy.optimize import brentq
from numpy.random import default_rng
colors = ['blue', 'orange']
def expand_data(data, std=0.1, seed=5):
    extended_data = pd.DataFrame(columns=['sxx', 'syy', 'classifier'])
    for i, row in data.iterrows():
        rng = default_rng(seed=seed)
        for norm in rng.normal(row['yieldstress'],
                                     std*row['yieldstress'],
                                     10):
            if norm < row['yieldstress']:
                add_data = pd.DataFrame.from_dict(
                    {'sxx': [row['sxx']*norm],
                     'syy': [row['syy']*norm],
                     'classifier': ['elastic']})
                extended_data = pd.concat([extended_data,
                    add_data],
                     ignore_index = True
                )
            if norm >= row['yieldstress']:
                add_data = pd.DataFrame.from_dict(
                    {'sxx': [row['sxx']*norm],
                     'syy': [row['syy']*norm],
                     'classifier': ['plastic']})
                extended_data = pd.concat([extended_data,
                    add_data],
                     ignore_index = True
                )
    return extended_data

def plot_train_pred_data(all_data, train_data=False,
                        test_data=False,
                        support_vectors=False,
                        pred_Y = False,
                        title = False):
    fig = go.Figure()
    
    if not isinstance(pred_Y, bool):
      fig.add_trace(go.Scatter(x = pred_Y['sYx'], 
                         y = pred_Y['sYy'],
                         mode = 'markers',
                         marker = dict(
                                  color = 'black'),
                         name = "Predicted Yield Stress"         
                                  ))

    
    
    if not isinstance(train_data, bool):
      for i,mode in enumerate(['elastic', 'plastic']):
          sub = train_data.where(train_data['classifier']==mode).dropna()
          fig.add_trace(go.Scatter(x=sub['sxx'],
                      y=sub['syy'],
                      mode='markers',
                      name = 'Train %s' %mode,
                      marker=dict(
                          color='white',
                          size=4,
                          line=dict(
                              color=colors[i],
                              width=2
                          )
                              )
                          )
                  )
    if not isinstance(test_data, bool):
      for i,mode in enumerate(['elastic', 'plastic']):
          sub = test_data.where(test_data['classifier_pred']==mode).dropna()
          fig.add_trace(go.Scatter(x=sub['sxx'],
                      y=sub['syy'],
                      mode='markers',
                      name = 'Test %s' %mode,
                      marker=dict(
                          color=colors[i],
                          size=4,
                          line=dict(
                              color=colors[i],
                              width=2
                          )
                              )
                          )
                      )
    if not isinstance(support_vectors, bool):
      fig.add_trace(go.Scatter(x=support_vectors[:,0],
                      y=support_vectors[:,1],
                      mode='markers',
                      name = 'Support Vectors',
                      marker=dict(
                          color='black',
                          size=4
                          )
                              )
                          )
    fig.add_trace(go.Scatter(x=all_data['sigYx'],
                            y=all_data['sigYy'],
                            mode='markers',
                            name = 'Yield Surface',
                            marker=dict(
                                color = 'green',
                                size = 4,
                                line=dict(
                            color='green',
                            width=2
                                )
                            )       
                            )
                        )
    if not isinstance(title, bool):
      fig.update_layout(
      title=title, 
      xaxis_title="sxx [MPa]",
      yaxis_title="syy [MPa]")

    fig.show()

def Multiplyer( X, multiplier):
    X2 = copy.deepcopy(X)
    X2 = X2*multiplier
    return X2 
    
def predict_YS_SVC(X_Test, clf_pipe):            
    '''Find transition from elastic to plastic deformation'''
    for i in range(len(X_Test)):
        def f(a, features, clf):
            feat = Multiplyer( features, a)
            return clf_pipe.decision_function(feat)
        upper_bound = 140
        lower_bound = 0
        c = brentq(f, lower_bound, upper_bound,
                    args=(X_Test, clf_pipe) , xtol=1e-4)
    return c
