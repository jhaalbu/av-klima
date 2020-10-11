import extreme as e
import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np

data1 = pd.read_csv('barsnes_3dogn_sno.txt')
#print(data1)
liste = data1.iloc[:, 0].tolist()
print(liste)
array = np.array(liste)
#data = ske.datasets.portpirie()

#print(data.description)

data_array = data1.values.tolist()
#print(data_array)
#sea_levels = data.fields.sea_level

#print(sea_levels)

#model = ske.models.engineering.Lieblein(array)
#array1000 = np.array([5000])
#print(array1000)
#model.plot_summary()
#plt.show()

#print(model.c, model.loc, model.scale)

#model2 = ske.models.classic.GEV(sea_levels, fit_method = 'mle', ci = 0.05, ci_method = 'delta', return_periods=5000)
#model2 = ske.models.classic.GEV(sea_levels, fit_method = 'mle')
#model2 = ske.models.classic.GEV(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
model2 = e.Gumbel(array, fit_method = 'mle', ci = 0.05, ci_method = 'delta')
#model.params

# #OrderedDict([('shape', 0.050109518363545352),
#              ('location', 3.8747498425529501),
#              ('scale', 0.19804394476624812)])

ax, value = model2.plot_return_values()
print(value)
plt.show()