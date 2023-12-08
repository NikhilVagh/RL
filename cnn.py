import numpy as np

from tensorflow.keras import layers

from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import cv2


# Function to add random noise to images
def add_noise(image_array, noise_factor=1):
    return image_array + noise_factor * np.random.normal(loc=0.0, scale=25.0, size=image_array.shape)/255

# Function to preprocess images
def preprocess_image(image_path, target_size=(256, 256)):
    img = load_img(image_path, color_mode='grayscale', target_size=target_size)
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to create and train the denoising autoencoder model
def train_denoising_autoencoder(train_images, noisy_train_images, epochs=50):
    input_img = Input(shape=(256, 256, 1))
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    encoded = MaxPooling2D((2, 2), padding='same')(x)

    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer="adam", loss='binary_crossentropy')



    # Train the model
    autoencoder.fit(x=noisy_train_images, y=train_images, epochs=epochs, batch_size=128, shuffle=True, validation_data=(noisy_train_images, train_images))
    return autoencoder

train_image_paths = list()

# with open('/home/abhishekj/nikhil/RL/training_BSD68.txt') as f:
#     [train_image_paths.append(line) for line in f.readlines()]

for line in open('/home/abhishekj/nikhil/RL/training_BSD68.txt'):
    train_image_paths.append(line.strip())
    # print(line.strip())

# print(train_image_paths)

# Load and preprocess training images
# train_image_paths = ['BSD68/gray/train/100007.jpg', 'BSD68/gray/train/100039.jpg', 'BSD68/gray/train/100075.jpg']
train_images = np.concatenate([preprocess_image(path) for path in train_image_paths])

noisy_train_image = add_noise(train_images)

# Train the denoising autoencoder model
denoising_autoencoder = train_denoising_autoencoder(train_images, noisy_train_image, epochs=100)

# Test the model on a new input image
input_image_path = 'BSD68/gray/train/100007.jpg'  # Replace with the path to your test image
input_image = preprocess_image(input_image_path)

# Add noise to the input image
noisy_input_image = add_noise(input_image)

# Denoise the image
denoised_output = denoising_autoencoder.predict(noisy_input_image)

# Original Noisy Image
cv2.imwrite('./temImage/noisy_input_image.png',(noisy_input_image[0, :, :, 0]*255).astype(np.uint8))

# Original Clean Image
print((input_image[0, :, :, 0]*255))
cv2.imwrite('./temImage/input_image.png',(input_image[0, :, :, 0]*255))

# Denoised Image
cv2.imwrite('./temImage/denoised_output.png',(denoised_output[0, :, :, 0]*255).astype(np.uint8))

print(np.mean(np.square(input_image[0, :, :, 0] - denoised_output[0, :, :, 0])))
