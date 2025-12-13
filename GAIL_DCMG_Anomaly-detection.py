from keras.optimizers import Adam
import numpy as np
from keras.layers import Input, Dense, Dropout, LSTM, RepeatVector, TimeDistributed
from keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib
import warnings
from sklearn.metrics import recall_score, roc_curve, auc
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neural_network import MLPClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras import regularizers
from keras.layers import Input, Dense
from keras.models import Model
from pandas.plotting import parallel_coordinates
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pandas as pd
from matplotlib import rcParams
from scipy.interpolate import interp1d
import seaborn as sns
import matplotlib.font_manager as font_manager
from sklearn.model_selection import train_test_split, GridSearchCV
from mpl_toolkits.axes_grid1 import inset_locator
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import numpy as np

def zone_and_linked(ax, axins, zone_left, zone_right, x, y, linked='bottom',
                    x_ratio=0.05, y_ratio=0.05):
    xlim_left = x[zone_left] - (x[zone_right] - x[zone_left]) * x_ratio
    xlim_right = x[zone_right] + (x[zone_right] - x[zone_left]) * x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data) - (np.max(y_data) - np.min(y_data)) * y_ratio
    ylim_top = np.max(y_data) + (np.max(y_data) - np.min(y_data)) * y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left, xlim_right, xlim_right, xlim_left, xlim_left],
            [ylim_bottom, ylim_bottom, ylim_top, ylim_top, ylim_bottom], "black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_left, ylim_bottom)
        xyA_2, xyB_2 = (xlim_right, ylim_top), (xlim_right, ylim_bottom)
    elif linked == 'top':
        xyA_1, xyB_1 = (xlim_left, ylim_bottom), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_right, ylim_top)
    elif linked == 'left':
        xyA_1, xyB_1 = (xlim_right, ylim_top), (xlim_left, ylim_top)
        xyA_2, xyB_2 = (xlim_right, ylim_bottom), (xlim_left, ylim_bottom)
    elif linked == 'right':
        xyA_1, xyB_1 = (xlim_left, ylim_top), (xlim_right, ylim_top)
        xyA_2, xyB_2 = (xlim_left, ylim_bottom), (xlim_right, ylim_bottom)

    con = ConnectionPatch(xyA=xyA_1, xyB=xyB_1, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2, xyB=xyB_2, coordsA="data",
                          coordsB="data", axesA=axins, axesB=ax)
    axins.add_artist(con)

warnings.filterwarnings('ignore')

plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.family'] = 'Times New Roman, SimSun'
plt.rcParams['mathtext.fontset'] = 'stix'

matplotlib.rcParams.update({'font.size': 22})

data = pd.read_csv('TrainData_GANData.csv')
data.dropna(subset=['label'], inplace=True)

voltage = data[['V_gu1', 'V_gu2', 'V_gu3', 'V_gu4']].values
current = data[['Idc1', 'Idc2', 'Idc3', 'Idc4']].values
power = data[['P_difference']].values
labels = data['label'].values

scaler = MinMaxScaler()
voltage_scaled = scaler.fit_transform(voltage)

fft_voltage = np.abs(np.fft.fft(voltage))
fft_current = np.abs(np.fft.fft(current))

mean_voltage = np.mean(voltage, axis=1)
std_voltage = np.std(voltage, axis=1)
mean_current = np.mean(current, axis=1)
std_current = np.std(current, axis=1)

X = np.column_stack((mean_voltage, std_voltage, mean_current, std_current,
                     fft_voltage[:, 1], fft_current[:, 1]))
y = labels

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

noise_level = 0.7
X_scaled = X_scaled + noise_level * np.random.normal(size=X_scaled.shape)
X_scaled = scaler.fit_transform(X_scaled)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

features = [f"Feature_{i}" for i in range(1, X_scaled.shape[1] + 1)]
data_scaled = pd.DataFrame(X_train, columns=features)
data_scaled['Target'] = y_train

feature_labels = ['Voltage Mean', 'Voltage Std', 'Current Mean', 'Current Std', 'Voltage Freq', 'Current Freq']

plt.figure(figsize=(12, 6))
for i, feature in enumerate(features):
    plt.subplot(2, 3, i + 1)
    sns.histplot(data_scaled[feature], kde=True, bins=20, color='blue')
    plt.xlabel(feature_labels[i], fontsize='medium')
    plt.ylabel('Count', fontsize='medium')
    plt.yticks(fontproperties='Times New Roman')
    plt.xticks(fontproperties='Times New Roman')
    plt.tick_params(direction='in')
plt.tight_layout()
plt.savefig('histogram_proposed.pdf', format='pdf', bbox_inches='tight')
plt.show()

sns.countplot(x='Target', data=data_scaled, palette='pastel')
plt.xlabel("Class")
plt.ylabel("Count")
plt.tick_params(direction='in')
plt.show()

counts = data_scaled['Target'].value_counts().reset_index()
counts.columns = ['Class', 'Count']
counts.to_csv('target_class_counts_forest.csv', index=False)

plt.figure(figsize=(7.5, 6))
correlation_matrix = data_scaled[features].corr()
sns.heatmap(correlation_matrix, annot=True, annot_kws={"color": "black"}, cmap='coolwarm', fmt='.2f', xticklabels=feature_labels, yticklabels=feature_labels)
ax = plt.gca()
ax.set_xticklabels(feature_labels, rotation=45)
plt.tick_params(direction='in')
plt.savefig('feature_correlation_matrix_proposed.pdf', format='pdf', bbox_inches='tight')
plt.show()

if not isinstance(correlation_matrix, pd.DataFrame):
    correlation_matrix = pd.DataFrame(correlation_matrix)

correlation_matrix.index = features
correlation_matrix.columns = features
correlation_matrix.to_csv('feature_correlation_matrix_proposed.csv', index=True, header=True)

model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

importances = model.feature_importances_
sorted_idx = np.argsort(importances)

plt.figure(figsize=(7.5, 3))
plt.plot(np.arange(len(features)), importances[sorted_idx], marker='o', linestyle='-', color='blue')
plt.xticks(ticks=np.arange(len(features)), labels=feature_labels, rotation=45)
plt.xlabel("Feature")
plt.ylabel("Importance Score")
plt.tick_params(direction='in')
plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
plt.xlim(0 - 0.5, len(features) - 0.5)
y_min = min(importances[sorted_idx])
y_max = max(importances[sorted_idx])
plt.ylim(0, 0.25)
plt.savefig('feature_importance_proposed.pdf', format='pdf', bbox_inches='tight')
plt.show()

feature_names = np.array(features)[sorted_idx]
importance_scores = importances[sorted_idx]

data = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importance_scores
})
data.to_csv('feature_importance_forest.csv', index=False)

y_pred = model.predict_proba(X_test)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [1, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42, class_weight='balanced'),
                           param_grid=param_grid,
                           cv=5,
                           scoring='accuracy',
                           verbose=1,
                           n_jobs=-1)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print("Best Parameters:", best_params)

best_rf_model = grid_search.best_estimator_
y_pred_optima = best_rf_model.predict_proba(X_test)

y_pred_class = np.argmax(y_pred_optima, axis=1)

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
n_classes = y_test_bin.shape[1]

accuracy = accuracy_score(y_test, y_pred_class)
precision = precision_score(y_test, y_pred_class, average='macro')
recall = recall_score(y_test, y_pred_class, average='macro')
f1 = f1_score(y_test, y_pred_class, average='macro')
weighted_precision = precision_score(y_test, y_pred_class, average='weighted')
weighted_recall = recall_score(y_test, y_pred_class, average='weighted')
weighted_f1 = f1_score(y_test, y_pred_class, average='weighted')
roc_auc = roc_auc_score(y_test_bin, y_pred_optima, average='macro')

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"ROC AUC: {roc_auc}")
print(f"Precision (Weighted): {weighted_precision}")
print(f"Recall (Weighted): {weighted_recall}")
print(f"F1 Score (Weighted): {weighted_f1}")

metrics = {
    'Accuracy': accuracy,
    'Precision (Macro)': precision,
    'Recall (Macro)': recall,
    'F1 Score (Macro)': f1,
    'ROC AUC': roc_auc,
    'Precision (Weighted)': weighted_precision,
    'Recall (Weighted)': weighted_recall,
    'F1 Score (Weighted)': weighted_f1
}
metrics_df = pd.DataFrame(list(metrics.items()), columns=['Metric', 'Value'])
metrics_df.to_csv('performance_metrics_proposed.csv', index=False)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred_optima[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test_bin.ravel(), y_pred_optima.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

fig = plt.figure(figsize=(8, 6))
plt.plot(fpr[0], tpr[0], color='blue', lw=2, label='Normal (AUC = %0.2f)' % roc_auc[0])
plt.plot(fpr[1], tpr[1], color='orange', lw=2, label='Attack (AUC = %0.2f)' % roc_auc[1])
plt.plot(fpr[2], tpr[2], color='green', lw=2, label='Fault (AUC = %0.2f)' % roc_auc[2])
plt.plot(fpr["micro"], tpr["micro"], color='red', lw=2, label='Micro-average (AUC = %0.2f)' % roc_auc["micro"])

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right", edgecolor='black', framealpha=1)
plt.grid(True, color='gray', linestyle='-', linewidth=0.5)
plt.tick_params(direction='in')

inset_ax = fig.add_axes([0.25, 0.55, 0.3, 0.2], facecolor="white")
inset_ax.plot(fpr[0], tpr[0], color='blue', lw=2)
inset_ax.plot(fpr[1], tpr[1], color='orange', lw=2)
inset_ax.plot(fpr[2], tpr[2], color='green', lw=2)
inset_ax.plot(fpr["micro"], tpr["micro"], color='red', lw=2)
inset_ax.set_xlim([0, 0.1])
inset_ax.set_ylim([0.9, 1])
inset_ax.grid()
plt.savefig('roc_proposed.pdf', format='pdf', bbox_inches='tight')
plt.show()

roc_curves_data = []
for key in fpr.keys():
    fpr_values = fpr[key]
    tpr_values = tpr[key]

    fpr_interp = np.linspace(0, 1, 100)
    tpr_interp = interp1d(fpr_values, tpr_values, kind='linear', fill_value="extrapolate")(fpr_interp)

    for i in range(len(fpr_interp)):
        roc_curves_data.append({
            'Class': key if key != "micro" else 'Micro-average',
            'FPR': fpr_interp[i],
            'TPR': tpr_interp[i]
        })

roc_curves_df = pd.DataFrame(roc_curves_data)
roc_curves_df.to_csv('roc_curves_proposed.csv', index=False)

conf_matrix = confusion_matrix(y_test, y_pred_class)
plt.figure(figsize=(7.5, 6))
im = plt.imshow(conf_matrix, cmap='Blues')
cbar = plt.colorbar(im, fraction=0.05)
tick_marks = np.arange(3)
plt.xticks(tick_marks, ['Normal', 'Attack', 'Fault'])
plt.yticks(tick_marks, ['Normal', 'Attack', 'Fault'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
thresh = conf_matrix.max() / 2.
for i in range(conf_matrix.shape[0]):
    for j in range(conf_matrix.shape[1]):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 ha="center", va="center",
                 color="black" if conf_matrix[i, j] > thresh else "black")
plt.tick_params(direction='in')
plt.savefig('conf_matrix_proposed.pdf', format='pdf', bbox_inches='tight')
plt.show()

conf_matrix_df = pd.DataFrame(conf_matrix, index=['Normal', 'Attack', 'Fault'], columns=['Normal', 'Attack', 'Fault'])
conf_matrix_df.to_csv('confusion_matrix_proposed.csv')

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

colors = {0: 'green', 1: 'orange', 2: 'blue'}

for label in colors.keys():
    if label == 0:
        ax.scatter(X_test[y_pred_class == label, 0],
                   X_test[y_pred_class == label, 2],
                   X_test[y_pred_class == label, 4],
                    color=colors[label], alpha=0.2, label='Normal', marker='o')
    elif label == 1:
        ax.scatter(X_test[y_pred_class == label, 0],
                   X_test[y_pred_class == label, 2],
                   X_test[y_pred_class == label, 4],
                    color=colors[label], alpha=0.2, label='Attack', marker='x')
    else:
        ax.scatter(X_test[y_pred_class == label, 0],
                   X_test[y_pred_class == label, 2],
                   X_test[y_pred_class == label, 4],
                    color=colors[label], alpha=0.2, label='Fault', marker='s')
ax.set_xlabel('Voltage Mean', labelpad=7)
ax.set_ylabel('Current Mean', labelpad=3)
ax.set_zlabel('Voltage Freq', labelpad=0)
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False
legend = plt.legend(title='', loc='lower left', bbox_to_anchor=(0.6, 0.6), edgecolor='black', framealpha=1)
plt.setp(legend.get_title(), visible=False)
ax.tick_params(axis='x', which='major', pad=0, direction='in')
ax.tick_params(axis='y', which='major', pad=0, direction='in')
ax.tick_params(axis='z', which='major', pad=0, direction='in')
font_path = font_manager.findfont('Times New Roman', fallback_to_default=False)
font_prop = font_manager.FontProperties(fname=font_path)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontproperties(font_prop)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontproperties(font_prop)
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontproperties(font_prop)
plt.savefig('3d_proposed1.pdf', format='pdf')
plt.show()

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

for label in colors.keys():
    if label == 0:
        ax.scatter(X_test[y_pred_class == label, 1],
                   X_test[y_pred_class == label, 3],
                   X_test[y_pred_class == label, 5],
                    color=colors[label], alpha=0.2, label='Normal', marker='o')
    elif label == 1:
        ax.scatter(X_test[y_pred_class == label, 1],
                   X_test[y_pred_class == label, 3],
                   X_test[y_pred_class == label, 5],
                    color=colors[label], alpha=0.2, label='Attack', marker='x')
    else:
        ax.scatter(X_test[y_pred_class == label, 1],
                   X_test[y_pred_class == label, 3],
                   X_test[y_pred_class == label, 5],
                    color=colors[label], alpha=0.2, label='Fault', marker='s')

ax.set_xlabel('Voltage Std', labelpad=7)
ax.set_ylabel('Current Std', labelpad=3)
ax.set_zlabel('Current Freq', labelpad=0)
ax.w_xaxis.pane.fill = False
ax.w_yaxis.pane.fill = False
ax.w_zaxis.pane.fill = False
legend = plt.legend(title='', loc='lower left', bbox_to_anchor=(0.6, 0.6), edgecolor='black', framealpha=1)
plt.setp(legend.get_title(), visible=False)
ax.tick_params(axis='x', which='major', pad=0, direction='in')
ax.tick_params(axis='y', which='major', pad=0, direction='in')
ax.tick_params(axis='z', which='major', pad=0, direction='in')
font_path = font_manager.findfont('Times New Roman', fallback_to_default=False)
font_prop = font_manager.FontProperties(fname=font_path)
for tick in ax.xaxis.get_major_ticks():
    tick.label.set_fontproperties(font_prop)
for tick in ax.yaxis.get_major_ticks():
    tick.label.set_fontproperties(font_prop)
for tick in ax.zaxis.get_major_ticks():
    tick.label.set_fontproperties(font_prop)
plt.savefig('3d_proposed2.pdf', format='pdf')
plt.show()