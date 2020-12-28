import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from tracer import Tracer


# Load data
df = pd.read_json('data.json')
print(df['id'].tolist())

# plt.figure(figsize=(10, 6))
# sns.scatterplot(x='latitude', y='longitude', data=df, hue='id')
# plt.legend(bbox_to_anchor=[1, 0.8])
# plt.show()

tracer = Tracer(df=df)

infected_people = tracer.get_infected_people(patient='Bob')
print(f'Infected Clusters: {infected_people}')
# plt.savefig('scatterplot.png', bbox_inches='tight')
