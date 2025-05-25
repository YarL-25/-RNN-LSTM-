import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Masking, SimpleRNN, Dense, Activation, Permute, Multiply, Lambda, RepeatVector, Flatten
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import ast
import joblib


df = pd.read_csv("final_dataset_rnn_ready.csv")
df = df.dropna(subset=['job_search_duration_months', 'career_sequence'])


def extract_sequence_for_model(seq_str):
    try:
        seq = ast.literal_eval(seq_str)
        return [hash(step['industry'] + '-' + step['position']) % 10000 for step in seq if 'industry' in step and 'position' in step]
    except:
        return []


df['sequence_encoded'] = df['career_sequence'].apply(extract_sequence_for_model)
X = pad_sequences(df['sequence_encoded'], maxlen=10, padding='post')
y = df['job_search_duration_months'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_layer = Input(shape=(10,))
embedding = Embedding(input_dim=10000, output_dim=64, input_length=10)(input_layer)
mask = Masking(mask_value=0.0)(embedding)
rnn_output = SimpleRNN(64, return_sequences=True)(mask)

attention_scores = Dense(1, activation='tanh')(rnn_output)
attention_scores = Flatten()(attention_scores)
attention_weights = Activation('softmax', name='attention_weights')(attention_scores)
attention_weights = RepeatVector(64)(attention_weights)
attention_weights = Permute([2, 1])(attention_weights)
attended_output = Multiply()([rnn_output, attention_weights])
context_vector = Lambda(lambda x: tf.reduce_sum(x, axis=1))(attended_output)

output = Dense(1)(context_vector)
model = Model(inputs=input_layer, outputs=output)
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary()

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)
model.save("rnn_attention_model.h5")
joblib.dump(history.history, "rnn_attention_history.pkl")

y_pred = model.predict(X_test).flatten()
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

results_df = pd.DataFrame({"MAE": [mae], "RMSE": [rmse], "R2": [r2]})
results_df.to_csv("rnn_attention_metrics.csv", index=False)

history_df = pd.DataFrame(history.history)

plt.figure(figsize=(10, 5))
plt.plot(history_df['loss'], label='Train Loss (MSE)')
plt.plot(history_df['val_loss'], label='Val Loss (MSE)')
plt.title("График функции потерь (MSE) по epochs")
plt.xlabel("epochs")
plt.ylabel("MSE")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(history_df['mae'], label='Train MAE')
plt.plot(history_df['val_mae'], label='Val MAE')
plt.title("Средняя абсолютная ошибка (MAE) по epochs")
plt.xlabel("epochs")
plt.ylabel("MAE")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 8))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel("Фактическое значение")
plt.ylabel("Предсказанное значение")
plt.title("Фактические vs. Предсказанные значения")
plt.grid(True)
plt.axis('equal')
plt.show()

residuals = y_test - y_pred
plt.figure(figsize=(10, 5))
sns.histplot(residuals, bins=30, kde=True)
plt.title("Распределение остатков (фактическое - предсказанное)")
plt.xlabel("Ошибка")
plt.ylabel("Количество")
plt.grid(True)
plt.show()

# === ШАГ 8: Визуализация attention-весов ===
# Создаем новую модель, возвращающую веса attention
attention_extractor = Model(inputs=model.input, outputs=model.get_layer('attention_weights').output)
attention_weights_test = attention_extractor.predict(X_test)

plt.figure(figsize=(12, 6))
plt.imshow(attention_weights_test[:10], cmap='viridis', aspect='auto')
plt.colorbar(label="Attention Weight")
plt.xlabel("Шаг последовательности")
plt.ylabel("Примеры из теста (первые 10)")
plt.title("Визуализация attention-весов по временным шагам")
plt.show()
