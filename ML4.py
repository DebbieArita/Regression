# Linear Regression
import pandas
df = pandas.read_csv('london_merged.csv')
print(df)
print(df.shape)
print(df.describe())


subset = df[['hum', 'wind_speed', 'cnt']]
array = subset.values
X = array[:, 0:2] # means all rows from columns 0...1
y = array[:, 2] # 9th is counted here

from sklearn import model_selection
X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, y, test_size=0.30, random_state=42)

# Linear regression
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR

model = LinearRegression()
model.fit(X_train, Y_train)
print('Learning completed!')

# ask the model to predict X_test
predictions = model.predict(X_test)
print(predictions)

# check accuracy/performance
from sklearn.metrics import r2_score
# r squared shows the percentage
print('R squared: ', r2_score(Y_test, predictions))

from sklearn.metrics import mean_squared_error
print('Mean squared error ', mean_squared_error(Y_test, predictions))
# above its squared, so we find square root

new = [[94.0, 8.0]]
observation = model.predict(new)
print('You will share ', observation, 'bikes')



# plot linear regression
import matplotlib.pyplot as plt
plt.style.use('seaborn')
fig, ax = plt.subplots()
ax.scatter(Y_test, predictions)
ax.plot(Y_test, Y_test)
ax.set_title('Predictions vs Y_test')
ax.set_xlabel('Y test')
ax.set_ylabel('Predictions')
plt.show()








