import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, BatchNormalization, LeakyReLU, Reshape, Flatten, \
    TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)


def load_and_preprocess_data(filepath):
    data = pd.read_csv(filepath)
    data.dropna(inplace=True)

    voltage_cols = ['V_gu1', 'V_gu2', 'V_gu3', 'V_gu4']
    current_cols = ['Idc1', 'Idc2', 'Idc3', 'Idc4']

    sequence_length = 10
    num_features = len(voltage_cols) + len(current_cols)

    X_normal = []
    X_attack = []
    X_fault = []

    for idx in range(len(data) - sequence_length + 1):
        sequence_data = data.iloc[idx:idx + sequence_length]
        voltage_seq = sequence_data[voltage_cols].values
        current_seq = sequence_data[current_cols].values
        seq_features = np.concatenate([voltage_seq, current_seq], axis=1)

        middle_idx = sequence_length // 2
        label = sequence_data.iloc[middle_idx]['label'] if 'label' in sequence_data.columns else 0

        if label == 0:
            X_normal.append(seq_features)
        elif label == 1:
            X_attack.append(seq_features)
        elif label == 2:
            X_fault.append(seq_features)

    X_normal = np.array(X_normal)
    X_attack = np.array(X_attack)
    X_fault = np.array(X_fault)

    return X_normal, X_attack, X_fault, sequence_length, num_features


def build_conditional_lstm_gan(latent_dim, sequence_length, num_features, num_classes):
    noise_input = Input(shape=(latent_dim,))
    label_input = Input(shape=(1,))

    label_embedding = Dense(latent_dim)(label_input)
    label_embedding = LeakyReLU(alpha=0.2)(label_embedding)

    merged = tf.keras.layers.multiply([noise_input, label_embedding])

    x = Dense(128)(merged)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Reshape((1, 128))(x)

    x = LSTM(128, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = LSTM(256, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    x = LSTM(256, return_sequences=True)(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = BatchNormalization(momentum=0.8)(x)

    output = TimeDistributed(Dense(num_features, activation='tanh'))(x)

    generator = Model([noise_input, label_input], output)

    sequence_input = Input(shape=(sequence_length, num_features))
    discriminator_label_input = Input(shape=(1,))

    label_embedding_d = Dense(sequence_length * num_features)(discriminator_label_input)
    label_embedding_d = Reshape((sequence_length, num_features))(label_embedding_d)

    merged_d = tf.keras.layers.concatenate([sequence_input, label_embedding_d])

    y = LSTM(128, return_sequences=True)(merged_d)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dropout(0.3)(y)

    y = LSTM(256, return_sequences=True)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dropout(0.3)(y)

    y = LSTM(256)(y)
    y = LeakyReLU(alpha=0.2)(y)
    y = Dropout(0.3)(y)

    discriminator_output = Dense(1, activation='sigmoid')(y)

    discriminator = Model([sequence_input, discriminator_label_input], discriminator_output)

    return generator, discriminator


def train_gan_for_anomaly_generation(X_normal, X_attack, X_fault,
                                     target_samples_per_class=1000,
                                     epochs=1000, batch_size=32):
    latent_dim = 100
    sequence_length = X_normal.shape[1]
    num_features = X_normal.shape[2]
    num_classes = 3

    generator, discriminator = build_conditional_lstm_gan(
        latent_dim, sequence_length, num_features, num_classes
    )

    discriminator.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5),
        metrics=['accuracy']
    )

    discriminator.trainable = False

    noise = Input(shape=(latent_dim,))
    label = Input(shape=(1,))
    generated_sequence = generator([noise, label])
    validity = discriminator([generated_sequence, label])
    gan = Model([noise, label], validity)

    gan.compile(
        loss='binary_crossentropy',
        optimizer=Adam(learning_rate=0.0002, beta_1=0.5)
    )

    X_minority = X_attack
    y_minority = np.ones((len(X_minority), 1))
    half_batch = batch_size // 2

    d_losses = []
    g_losses = []

    for epoch in range(epochs):
        idx = np.random.randint(0, len(X_minority), half_batch)
        real_sequences = X_minority[idx]
        real_labels = np.ones((half_batch, 1))

        noise = np.random.normal(0, 1, (half_batch, latent_dim))
        gen_labels = np.ones((half_batch, 1))
        generated_sequences = generator.predict([noise, gen_labels], verbose=0)
        fake_labels = np.zeros((half_batch, 1))

        d_loss_real = discriminator.train_on_batch([real_sequences, gen_labels], real_labels)
        d_loss_fake = discriminator.train_on_batch([generated_sequences, gen_labels], fake_labels)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        valid_y = np.ones((batch_size, 1))
        gen_labels = np.ones((batch_size, 1))

        g_loss = gan.train_on_batch([noise, gen_labels], valid_y)

        d_losses.append(d_loss[0])
        g_losses.append(g_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: D loss: {d_loss[0]:.4f}, G loss: {g_loss:.4f}")

    return generator, discriminator, d_losses, g_losses


def extract_features_from_sequences(sequences):
    n_samples = sequences.shape[0]
    n_features = sequences.shape[2]

    extracted_features = []

    for i in range(n_samples):
        seq_features = []
        for j in range(n_features):
            feature_data = sequences[i, :, j]
            seq_features.extend([
                np.mean(feature_data),
                np.std(feature_data),
                np.min(feature_data),
                np.max(feature_data),
                np.median(feature_data),
                np.percentile(feature_data, 25),
                np.percentile(feature_data, 75),
            ])

        for j in range(n_features):
            feature_data = sequences[i, :, j]
            fft_result = np.abs(np.fft.fft(feature_data))
            seq_features.extend([
                fft_result[1],
                fft_result[2],
            ])

        extracted_features.append(seq_features)

    return np.array(extracted_features)


def generate_balanced_dataset(X_normal, X_attack, X_fault, generator,
                              target_samples_per_class=1000, latent_dim=100):
    num_normal = len(X_normal)
    num_attack = len(X_attack)
    num_fault = len(X_fault)

    if num_attack < target_samples_per_class:
        num_to_generate = target_samples_per_class - num_attack
        noise = np.random.normal(0, 1, (num_to_generate, latent_dim))
        labels = np.ones((num_to_generate, 1))
        generated_attack = generator.predict([noise, labels])
    else:
        generated_attack = X_attack[:target_samples_per_class]

    if num_fault < target_samples_per_class:
        num_to_generate = target_samples_per_class - num_fault
        noise = np.random.normal(0, 1, (num_to_generate, latent_dim))
        labels = np.full((num_to_generate, 1), 2)
        generated_fault = generator.predict([noise, labels])
    else:
        generated_fault = X_fault[:target_samples_per_class]

    if num_normal > target_samples_per_class:
        indices = np.random.choice(num_normal, target_samples_per_class, replace=False)
        X_normal_balanced = X_normal[indices]
    else:
        X_normal_balanced = X_normal

    X_balanced = np.concatenate([
        X_normal_balanced,
        generated_attack if 'generated_attack' in locals() else X_attack[:target_samples_per_class],
        generated_fault if 'generated_fault' in locals() else X_fault[:target_samples_per_class]
    ], axis=0)

    y_balanced = np.concatenate([
        np.zeros(len(X_normal_balanced)),
        np.ones(len(generated_attack) if 'generated_attack' in locals() else min(target_samples_per_class, num_attack)),
        np.full(len(generated_fault) if 'generated_fault' in locals() else min(target_samples_per_class, num_fault), 2)
    ])

    X_features = extract_features_from_sequences(X_balanced)

    return X_features, y_balanced


def save_balanced_dataset(X_features, y_balanced, output_filename='TrainData_GANData.csv'):
    n_stat_features = 7
    n_freq_features = 2
    n_original_features = X_features.shape[1] // (n_stat_features + n_freq_features)

    feature_names = []
    for i in range(n_original_features):
        base_name = f"Feature_{i + 1}"
        feature_names.extend([
            f"{base_name}_mean",
            f"{base_name}_std",
            f"{base_name}_min",
            f"{base_name}_max",
            f"{base_name}_median",
            f"{base_name}_q25",
            f"{base_name}_q75",
        ])

    for i in range(n_original_features):
        base_name = f"Feature_{i + 1}"
        feature_names.extend([
            f"{base_name}_freq1",
            f"{base_name}_freq2",
        ])

    df = pd.DataFrame(X_features, columns=feature_names)
    df['label'] = y_balanced
    df.to_csv(output_filename, index=False)

    print(f"平衡数据集已保存到 {output_filename}")
    print(f"数据集形状: {df.shape}")
    print(f"各类样本数量:")
    print(f"  正常类 (0): {np.sum(y_balanced == 0)}")
    print(f"  攻击类 (1): {np.sum(y_balanced == 1)}")
    print(f"  故障类 (2): {np.sum(y_balanced == 2)}")

    return df


def main():
    print("加载原始数据...")
    X_normal, X_attack, X_fault, sequence_length, num_features = load_and_preprocess_data('your_original_data.csv')

    print(f"数据统计:")
    print(f"  正常类样本数: {len(X_normal)}")
    print(f"  攻击类样本数: {len(X_attack)}")
    print(f"  故障类样本数: {len(X_fault)}")
    print(f"  序列长度: {sequence_length}")
    print(f"  特征数: {num_features}")

    print("\n训练LSTM-GAN生成异常样本...")
    generator, discriminator, d_losses, g_losses = train_gan_for_anomaly_generation(
        X_normal, X_attack, X_fault,
        target_samples_per_class=1000,
        epochs=1000,
        batch_size=32
    )

    print("\n生成平衡数据集...")
    X_balanced, y_balanced = generate_balanced_dataset(
        X_normal, X_attack, X_fault, generator,
        target_samples_per_class=1000
    )

    print("\n保存平衡数据集...")
    save_balanced_dataset(X_balanced, y_balanced, 'TrainData_GANData.csv')

    plt.figure(figsize=(10, 5))
    plt.plot(d_losses, label='Discriminator Loss')
    plt.plot(g_losses, label='Generator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('LSTM-GAN Training Progress')
    plt.legend()
    plt.grid(True)
    plt.savefig('gan_training_progress.png', dpi=300, bbox_inches='tight')
    plt.show()



if __name__ == "__main__":
    main()