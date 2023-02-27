import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

# 设置plotly默认主题
pio.templates.default = 'plotly_white'

# 设置pandas打印时显示所有列
pd.set_option('display.max_columns', None)